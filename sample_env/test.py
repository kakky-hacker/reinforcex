import grpc
from concurrent.futures import ThreadPoolExecutor

import env_server_pb2
import env_server_pb2_grpc


def run_episode(server_idx):
    with grpc.insecure_channel(f"localhost:{8000 + server_idx}") as channel:
        stub = env_server_pb2_grpc.EnvServiceStub(channel)

        reset_response = stub.Reset(env_server_pb2.ResetRequest(env="CartPole-v1"))
        session_id = reset_response.session_id
        observation = list(reset_response.observation)
        print(f"[Thread {server_idx}] Reset: session_id={session_id}, obs={observation}")

        step_response = stub.Step(env_server_pb2.StepRequest(session_id=session_id, action=0))
        print(f"[Thread {server_idx}] Step: reward={step_response.reward}, done={step_response.done}")

        return step_response


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
