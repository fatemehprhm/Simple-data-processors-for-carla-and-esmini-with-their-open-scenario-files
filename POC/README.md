# Running a simulation
In order to run a specific scenario just change the simualation number in Run.sh file to sth like "sim1" or "sim2", etc:
```
python3 data_extractor.py --filename 'sim?' &

```
Then change the scenario directory in this snipset:
```
python3 scenario_runner.py --openscenario /home/argos/POC/Simulations/sim2/FollowLeadingVehicle_sim2.xosc --reloadWorld
```