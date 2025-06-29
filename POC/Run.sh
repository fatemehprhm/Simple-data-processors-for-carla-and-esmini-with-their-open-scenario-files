# Running Carla server
#!/bin/sh
UE4_TRUE_SCRIPT_NAME=$(echo \"$0\" | xargs readlink -f)
UE4_PROJECT_ROOT=$(dirname "$UE4_TRUE_SCRIPT_NAME")
chmod +x "$UE4_PROJECT_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
"$UE4_PROJECT_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 "$@" &

# Wait for the server to start
sleep 5

# Change the directory so the scenario runner works perfectly
cd "scenario_runner-0.9.13"

# Run the data exctractor
python3 data_extractor.py --filename 'sim2' &

# Wait for the scenario runner to start
sleep 5

# Run the main scenario
python3 scenario_runner.py --openscenario /home/argos/POC/Simulations/sim2/FollowLeadingVehicle_sim2.xosc --reloadWorld