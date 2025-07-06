from concurrent.futures import ThreadPoolExecutor

import requests


def run_episode(server_idx):
    res = requests.post(
        f"http://localhost:800{server_idx}/reset",
        json={"env": "CartPole-v1"},
    )
    data = res.json()
    session_id = data["session_id"]
    obs = data["observation"]
    print(f"Reset: {data}")
    res = requests.post(
        f"http://localhost:800{server_idx}/step",
        json={"session_id": session_id, "action": 0},
    )
    result = res.json()
    print(f"Step: {result}")

    return result


if __name__ == "__main__":
    num_threads = 10

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(run_episode, server_idx=i + 1) for i in range(num_threads)
        ]
        for future in futures:
            try:
                result = future.result()
            except Exception as e:
                print(f"Error in thread: {e}")
