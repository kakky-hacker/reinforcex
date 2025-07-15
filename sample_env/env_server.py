import uuid
import gymnasium as gym
import grpc
from concurrent import futures

import env_server_pb2
import env_server_pb2_grpc


environments = {}


class EnvService(env_server_pb2_grpc.EnvServiceServicer):
    def Reset(self, request, context):
        env = gym.make(request.env)
        obs, _ = env.reset()
        session_id = str(uuid.uuid4())
        environments[session_id] = env
        return env_server_pb2.ResetResponse(
            session_id=session_id,
            observation=obs.tolist()
        )

    def Step(self, request, context):
        session_id = request.session_id
        action = request.action

        if session_id not in environments:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Session ID not found")
            return env_server_pb2.StepResponse()

        env = environments[session_id]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            env.close()
            del environments[session_id]

        return env_server_pb2.StepResponse(
            observation=obs.tolist(),
            reward=reward,
            done=done,
            info={k: str(v) for k, v in info.items()}
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    env_server_pb2_grpc.add_EnvServiceServicer_to_server(EnvService(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
