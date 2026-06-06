import os
import torch
import cv2
import gymnasium as gym
import AutoROM
import av
import numpy as np
from torch.distributions.categorical import Categorical

from ppo import Actor, Backbone


ACTOR_PATH = "runs/rnd_pure/126141.pth"

# ROM setup for Atari environment
autorom_dir = os.path.dirname(AutoROM.__file__)
rom_dir = os.path.join(autorom_dir, "roms")
os.environ["ALE_ROMS_DIR"] = rom_dir


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Actor
    n_actions = 4
    actor = Actor(n_actions).to(device)

    if os.path.exists(ACTOR_PATH):
        print(f"Loading actor from {ACTOR_PATH}")
        actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    else:
        print(f"Warning: {ACTOR_PATH} not found. Using uninitialized actor for demonstration.")

    actor.eval()

    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    obs, info = env.reset()
    obs_buffer = [obs, obs]

    height, width, _ = obs.shape
    video_path = "playback.mkv"

    # Use av instead of cv2.VideoWriter
    container = av.open(video_path, "w", "matroska")
    stream = container.add_stream("libx264", rate=60)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "10", "preset": "ultrafast"}

    # Agent State
    state = Backbone.new_state(1).to(device)
    next_done = torch.zeros((1,), dtype=torch.bool, device=device)
    done = False
    episodic_reward = 0.0

    print(f"Starting playback. Recording to {video_path}...")

    try:
        while not done:
            obs_input = np.maximum(obs_buffer[0], obs_buffer[1])
            obs_input = cv2.cvtColor(obs_input, cv2.COLOR_RGB2GRAY)
            obs_input = cv2.resize(obs_input, (64, 64), interpolation=cv2.INTER_AREA)
            obs_input = torch.from_numpy(obs_input).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                state, logits, _, _ = actor.forward_single_step(state, obs_input, next_done)
                action = Categorical(logits=logits).sample().item()

            for _ in range(4):
                frame = av.VideoFrame.from_ndarray(obs, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

                obs, reward, terminated, truncated, info = env.step(action)
                obs_buffer[0] = obs_buffer[1]
                obs_buffer[1] = obs

                done = terminated or truncated
                episodic_reward += float(reward)

                if done:
                    frame = av.VideoFrame.from_ndarray(obs, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    break

    finally:
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        env.close()
        print(f"Playback finished. Video saved to {video_path}")
        print(f"Episodic reward: {episodic_reward}")


if __name__ == "__main__":
    play()
