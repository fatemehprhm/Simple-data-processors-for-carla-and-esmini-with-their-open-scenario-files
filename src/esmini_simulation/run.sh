#!/bin/bash

# Function to print usage instructions
usage() {
    echo "Usage: $0 --osc <scenario_address> --timestep <timestep> --file <file_name>"
}

# Check if the number of arguments is correct
if [ "$#" -ne 6 ]; then
    usage
    exit 1
fi

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --osc)
            scenario_address="$2"
            shift
            ;;
        --timestep)
            timestep="$2"
            shift
            ;;
        --file)
            file_name="$2"
            shift
            ;;
        *)
            # Invalid option
            usage
            exit 1
            ;;
    esac
    shift
done

if [ -z "$scenario_address" ] || [ -z "$timestep" ] || [ -z "$file_name" ]; then
    usage
    exit 1
fi

# Run the esmini command
cd ~/esmini

./bin/esmini --window 60 60 800 400 --osc "$scenario_address" --fixed_timestep "$timestep" --record "$file_name.dat"

#sleep 10
# Run the Python script
python3 scripts/dat2csv.py "$file_name.dat"
# python3 scripts/plot_dat.py "$file_name.dat'" --param speed

cd ~/alks_simulation/src/processors
python3 cutin_processor.py --csv "~/esmini/$file_name.csv" --output "$file_name.csv"