#!/bin/bash

if [ ! -f /data/merged.osm.pbf ]; then
    echo "Installing dependencies..."
    apt-get update && apt-get install -y wget osmium-tool ca-certificates

    echo "Downloading OSM files..."
    wget -O /data/belgium.osm.pbf https://download.geofabrik.de/europe/belgium-latest.osm.pbf
    wget -O /data/oberfranken.osm.pbf https://download.geofabrik.de/europe/germany/bayern/oberfranken-latest.osm.pbf
    wget -O /data/freiburg.osm.pbf https://download.geofabrik.de/europe/germany/baden-wuerttemberg/freiburg-regbez-latest.osm.pbf

    echo "Merging OSM files..."
    osmium merge /data/belgium.osm.pbf /data/oberfranken.osm.pbf /data/freiburg.osm.pbf -o /data/merged.osm.pbf --overwrite

    echo "Cleaning up..."
    rm /data/belgium.osm.pbf /data/oberfranken.osm.pbf /data/freiburg.osm.pbf
    echo "Done!"
else
    echo "Merged OSM file already exists. Skipping download and merge."
fi

# Execute the original entrypoint of the nominatim image
exec /app/start.sh "$@"
