import array
import math
import os
import struct
from datetime import datetime

import numpy as np

from OXTS import OXTS


class OxtsDataReader:

    root = "UNDEFINED"
    currentName = 0

    datamap = {}
    offsetmap = {}
    rotationMap = {}


    OXTS_DATA = "//oxts//data//"
    TIMESTAMPS = "//oxts//timestamps.txt"


    def getRoot(self, path):
        files = os.listdir(path)
        if len(files) == 1:
            return self.getRoot(path + "//" + files[0])
        else:
            return path


    def __init__(self, path):
        self.root = self.getRoot(path)

        count = 0

        timestamps = self.readTimestamps(self.root + self.TIMESTAMPS)


        for filename in os.listdir(self.root + self.OXTS_DATA):

            if filename.endswith(".txt"):
                self.readOx(filename, timestamps[count])
                count += 1

        if not len(self.datamap.keys()) == len(timestamps):
            raise RuntimeError("Timestamps and farmes do not match: keys[" + str(len(self.datamap.keys())) + "] timestamps[" + str(len(timestamps)) + "]")
        else:
            print("Frames and timestamps loaded")

        firstOxt = self.datamap[list(self.datamap.keys())[0]]  # get position and rotation origin
        gpsOrigin = self.getCartesian(firstOxt)
        rotOrigin = self.readRotation(firstOxt)

        for k in self.datamap.keys():  # prepare oxts data for usage
            self.offsetmap[k] = self.getCartesian(self.datamap[k]) - gpsOrigin
            self.rotationMap[k] = self.readRotation(self.datamap[k]) - rotOrigin

    def readTimestamps(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]

        date_format = "%Y-%m-%d %H:%M:%S.%f"  # Format string to match the input format
        timestamps = []
        for i in range(len(lines)):
            time = lines[i]
            time = time.split('.')
            time = time[0] + "." + time[1][:6]
            timestamps.append( datetime.strptime(time, date_format) )

        return timestamps

    def readOx(self, filename, timestamp):
        with open(self.root + self.OXTS_DATA + filename, 'r') as file:
            line = file.readline()
            line = line.strip()
            data = line.split(' ')
            data = np.array(data).astype(np.float64)


            filename = filename.strip('.txt')
            oxts = OXTS(
                latitude=data[0],
                longitude=data[1],
                altitude=data[2],
                roll=data[3],
                pitch=data[4],
                yaw=data[5],
                vel_north=data[6],
                vel_east=data[7],
                vel_forward=data[8],
                vel_left=data[9],
                vel_up=data[10],
                acc_x=data[11],
                acc_y=data[12],
                acc_z=data[13],
                acc_forward=data[14],
                acc_left=data[15],
                acc_up=data[16],
                angvel_x=data[17],
                angvel_y=data[18],
                angvel_z=data[19],
                angvel_forward=data[20],
                angvel_left=data[21],
                angvel_up=data[22],
                pos_accuracy=data[23],
                vel_accuracy=data[24],
                navstat=data[25],
                numsats=data[26],
                posmode=data[27],
                velmode=data[28],
                orimode=data[29],
                time=timestamp
            )
            self.datamap[filename] = oxts

    def getOxtsPosNp(self):
        points = []
        for k in self.datamap.keys():
            points.append(self.offsetmap[k])

        return np.array(points)

    def getCartesian(self, oxts):
        """
        x = R * cos(lat) * cos(lon)
        y = R * cos(lat) * sin(lon)
        z = R *sin(lat)
        """

        R = 6371000  # m
        lon_lat_alt = oxts.getWorldPos()


        lat = math.radians(lon_lat_alt[1])
        lon = math.radians(lon_lat_alt[0])
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        return np.mod(np.array([x, y, z]), 10000)

    def readRotation(self, oxts):
        return np.array([oxts.roll, oxts.pitch, oxts.yaw])

    def getRotation(self, key):
        return self.rotationMap[key]

    def getCurrentName(self):
        return self.getfilenames()[self.currentName]

    def getOx(self, filename):
        return self.datamap[filename]

    def getfilenames(self):
        return list(self.datamap.keys())

    def peekNextName(self):
        if self.currentName < len(self.datamap.keys()) - 1:
            return self.getfilenames()[self.currentName + 1]
        else:
            return UNDEFINED

    def getOriginOx(self):
        return self.datamap[ list(self.datamap.keys())[0] ]

    def getNextName(self):
        names = self.getfilenames()



        ret = names[self.currentName]
        self.currentName += 1
        if self.currentName >= len(names):
            self.currentName = 0
        return ret

    def getCurrentOx(self):
        return self.getOx(self.getCurrentName())

    def getWorldOrigin(self):
        firstName = self.getfilenames()[0]
        ox = self.getOx(firstName)
        return ox.getWorldPos()
