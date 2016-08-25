import subprocess 
import sys
from json import loads
from time import time
##
### rsync -avz eswanson@dpu-soa-ingest01:'$(find /tenant-data/12/aep_provided_data/*.txt -ctime -n)' .
### Note: -ctime -n is past n*24 hours or n days
##
### AEP Indiana & Michigan
aep0 = subprocess.run('scp eswanson@dpu-soa-ingest01:/tenant-data/12/aep_provided_data/*201607*  bills/',
           shell=True, check=True, stdout=subprocess.PIPE)


aep1 = subprocess.run('scp eswanson@dpu-soa-ingest01:/tenant-data/12/aep_provided_data/*201608*  bills/',
           shell=True, check=True, stdout=subprocess.PIPE)
print(aep0.stdout.decode('utf-8'))
print(aep1.stdout.decode('utf-8'))

# PPL
ppl0= subprocess.run('scp eswanson@dpu-soa-ingest01:/tenant-data/43/error/*1607*  bills/', shell=True, check=True, stdout=subprocess.PIPE)


ppl1 = subprocess.run('scp eswanson@dpu-soa-ingest01:/tenant-data/43/error/*1608*  bills/', shell=True, check=True, stdout=subprocess.PIPE)
print(ppl0.stdout.decode('utf-8'))
print(ppl1.stdout.decode('utf-8'))

# NV Energy North (SPPC) & South (NPC)
nve0 = subprocess.run('scp eswanson@dpu-soa-ingest01:/tenant-data/70132650159973376/error/*201607* bills/',
           shell = True, check = True, stdout = subprocess.PIPE)

nve1 = subprocess.run('scp eswanson@dpu-soa-ingest01:/tenant-data/70132650159973376/error/*201608* bills/',
           shell = True, check = True, stdout = subprocess.PIPE)
print(nve0.stdout.decode('utf-8'))
print(nve1.stdout.decode('utf-8'))

##
aep = subprocess.run('aws s3 cp bills/ s3://sciencebox-bills/MeasureValidate/AEP/ --recursive --exclude "*" --include "*aep*" ', shell=True, check=True, stdout=subprocess.PIPE)
print(aep.stdout.decode('utf-8'))

nvenergy = subprocess.run('aws s3 cp bills/ s3://sciencebox-bills/MeasureValidate/NVNorth/ --recursive --exclude "*" --include "*SPPC*"', shell=True, check=True, stdout=subprocess.PIPE)
print(nvenergy.stdout.decode('utf-8'))

nvenergy = subprocess.run('aws s3 cp bills/ s3://sciencebox-bills/MeasureValidate/NVSouth/ --recursive --exclude "*" --include "*NPC*"', shell=True, check=True, stdout=subprocess.PIPE)
print(nvenergy.stdout.decode('utf-8'))

ppl = subprocess.run('aws s3 cp ../201604_Extract_Bills_Experian/ s3://sciencebox-bills/MeasureValidate/PPL/ --recursive --exclude "*" --include "*Ecova*"', shell=True, check=True, stdout=subprocess.PIPE)
print(ppl.stdout.decode('utf-8'))


