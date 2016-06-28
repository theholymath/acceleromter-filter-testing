# acceleromters
The is code to test filters on phone accelerometer data and to visualize the data.

## accelerometer-filter-testing.py 
This file simply applies various filters and looks at the output. Kalman, moving average and butterworth are included.

## sensorstream_live_stream_acceleration.py
This is a visualization script. It uses the Andrid app found here: https://play.google.com/store/apps/details?id=de.lorenz_fenster.sensorstreamgps&hl=en and live streams
* raw acceleration data,
* filtered accleration data, and 
* Either the G-forces applied, or the tilt angles. 

Enjoy!
