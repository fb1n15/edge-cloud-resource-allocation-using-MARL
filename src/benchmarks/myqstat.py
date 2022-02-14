#!/opt/bin/python3
import os
import xml.etree.ElementTree as ET

#Fields
fields=['Job_Id','Job_Name','Job_Owner','job_state']
names=['Job ID','Job Name','Owner','Status']

#Get job info
f = os.popen('qstat -x')
tree = ET.parse(f)
root = tree.getroot()
n_fields=len(fields)
jobs=[[job.find(field).text for field in fields] for job in root]
max_lengths=[len(name) for name in names]
sep='  '

#Identify max characer length per field
for j in jobs:
    for i in range(n_fields):
            #Chop off anything after and including '@' or '.' from all fields
            if j[i].find('@')>0:
                    j[i]=j[i][:j[i].find('@')]
            if j[i].find('.')>0:
                    j[i]=j[i][:j[i].find('.')]
            if(len(j[i])>max_lengths[i]):
                    max_lengths[i]=len(j[i])

#Field names
for i in range(n_fields):
    print('{s:^{length}}'.format(s=names[i],length=max_lengths[i]),end=sep)
print()

#Dashes
for i in range(n_fields):
    print('-'*max_lengths[i],end=sep)
print()

#Jobs
for j in jobs:
    for i in range(n_fields):
            if j[i].find('@')>0:
                    j[i]=j[i][:j[i].find('@')]
            print('{s:<{length}}'.format(s=j[i],length=max_lengths[i]),end=sep)
    print()
