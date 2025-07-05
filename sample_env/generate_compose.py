import yaml

NUM_CONTAINERS = 10
services = {}

for i in range(1, NUM_CONTAINERS + 1):
    service_name = f"rl_env_{i}"
    services[service_name] = {
        "build": ".",
        "ports": [f"{8000 + i}:8000"],
    }

compose = {
    "version": "3",
    "services": services,
}

with open("docker-compose.yml", "w") as f:
    yaml.dump(compose, f, sort_keys=False)
