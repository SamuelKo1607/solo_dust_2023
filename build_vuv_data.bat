@echo off

echo %time%

call activate dust6

(
	start python 003_solar_orbiter/solo_vuv.py 12 0
	start python 003_solar_orbiter/solo_vuv.py 12 1
	start python 003_solar_orbiter/solo_vuv.py 12 2
	start python 003_solar_orbiter/solo_vuv.py 12 3
	start python 003_solar_orbiter/solo_vuv.py 12 4
	start python 003_solar_orbiter/solo_vuv.py 12 5
	start python 003_solar_orbiter/solo_vuv.py 12 6
	start python 003_solar_orbiter/solo_vuv.py 12 7
	start python 003_solar_orbiter/solo_vuv.py 12 8
	start python 003_solar_orbiter/solo_vuv.py 12 9
	start python 003_solar_orbiter/solo_vuv.py 12 10
	start python 003_solar_orbiter/solo_vuv.py 12 11
) | set /P "="

echo %time%

start python 003_solar_orbiter/solo_vuv_aggregate.py

echo %time%