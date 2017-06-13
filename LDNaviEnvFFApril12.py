from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import os.path
from collections import deque
import matplotlib.pyplot as plt

# Robot-centric environment. Use 5 dims to decribe obstacles. Companion is considered.
# This implements a goal-switching state formulation with 0/1 companion
# Easier no Comp scenario
# no LostCom terminal condition
# MIN_COM_DIST = 0.4

DATA_LIST = [\
             # './data/ewap_dataset_full/seq_eth/', \
             './data/ewap_dataset_full/seq_hotel/', \
             './data/socially compliant/',\
             './data/zara01/', \
             './data/zara02/', \
             './data/zara03/', \
             './data/ucy03/', \
             './data/ucy01/'
]


BOUND_OF_SPACE = 20# Maximum space = 10m*10m
BOUND_OF_OMEGA = 1  # Maximum Omega = 18 degrees/s
NUMBER_OF_PED = 3
NUMBER_OF_COM = 1
BOUND_OF_V = 1
REACH_TARGET_BOUND = 0.8
STATE_DIM = 13+2*NUMBER_OF_PED+2*NUMBER_OF_COM
STACK_FRAMES = 4
MAX_OBS_DISTANCE = 4
MAX_OBS_TRUE_DISTANCE = 10
MAX_PED_DISTANCE = 4
T = 0.1

MIN_PED_DIST = 0.5
MIN_OBS_DIST = 0.2
PED_DETECT_FOV = 240.0/180.0*np.pi
OBS_DETECT_FOV = 240.0/180.0*np.pi
MIN_COM_DIST = 0.4
MAX_COM_DIST = 2
MIN_OBS_DIST_MARGIN = 0.1
MIN_PED_DIST_MARGIN = 0.1
GOAL_SWITCHING_BOUND=[MAX_COM_DIST,MAX_COM_DIST]
PED_DIST_ALARM_BOUND = 1
DESIRED_COM_DISTANCE = 0.8
MAX_VT = 0.7

GOD_TIME = 5

class LDNaviEnvFFApril12(Env):
    @property
    def observation_space(self):
        space_low = np.ones(STATE_DIM*STACK_FRAMES,)*(-BOUND_OF_SPACE)
        for i in xrange(0,STACK_FRAMES):
            space_low[1+i*STATE_DIM] = -2*np.pi
        space_high = np.ones(STATE_DIM*STACK_FRAMES,) * (BOUND_OF_SPACE)
        for i in xrange(0,STACK_FRAMES):
            space_high[1+i*STATE_DIM] = 2*np.pi
        return Box(low=space_low, high=space_high)

    @property
    def action_space(self):
        return Box(low=np.array([-BOUND_OF_V, -BOUND_OF_OMEGA]), high=np.array([BOUND_OF_V,BOUND_OF_OMEGA]))

    def __init__(self):
        self.plotCount = 1

    def reset(self):
        n1 = np.random.randint(0,len(DATA_LIST))
        FN = DATA_LIST[n1]
        comp = np.random.rand()
        if comp>0.5:
            self.compMode = True
        else:
            self.compMode = False

        self.trajectoryX = []
        self.trajectoryY = []
        self.pedTrajX = [[],[],[],[],[]]
        self.pedTrajY = [[],[],[],[],[]]
        self.des = []
        self.comTrajX= [[]]
        self.comTrajY = [[]]
        self.dist2Com= 0.0
        self.obsCor = np.array([])
        self.trackV = []
        self.trackOmega = []
        self._t = 0

        self.hasLostCom = False
        self.maxComDist = DESIRED_COM_DISTANCE

        self._state = np.zeros(STATE_DIM)
        self._stack = deque([])
        self.trueState = np.zeros(STATE_DIM)
        self.trueStack = deque([])

        if FN == './data/socially compliant/':
            self.FPS = 0.1
            fileID = np.random.randint(2, 86)
            self.frameID = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ', dtype=float, usecols=(0,))
            self.pedID = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ', dtype=float, usecols=(1,))
            self.pedPosX = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ', dtype=float, usecols=(2,))
            self.pedPosY = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ', dtype=float, usecols=(3,))
            self.pedVolX = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ', dtype=float, usecols=(4,))
            self.pedVolY = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ', dtype=float, usecols=(5,))

            self.clipLength = self.frameID.size

            self.robot = np.random.randint(1, 5)

            self.gtTrajIndex = np.argwhere(self.pedID == self.robot)

            if self.gtTrajIndex.size == 0:
                distToGoal = 0
            else:
                self.des = [self.pedPosX[self.gtTrajIndex[-1]], self.pedPosY[self.gtTrajIndex[-1]]]
                distToGoal = np.sqrt((self.pedPosX[self.gtTrajIndex[0]] - self.des[0]) ** 2 + (
                    self.pedPosY[self.gtTrajIndex[0]] - self.des[1]) ** 2)

            while self.gtTrajIndex.size < 20 or distToGoal < 5:
                fileID = np.random.randint(2, 86)
                self.frameID = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ',
                                          dtype=float, usecols=(0,))
                self.pedID = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ',
                                        dtype=float, usecols=(1,))
                self.pedPosX = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ',
                                          dtype=float, usecols=(2,))
                self.pedPosY = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ',
                                          dtype=float, usecols=(3,))
                self.pedVolX = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ',
                                          dtype=float, usecols=(4,))
                self.pedVolY = np.loadtxt(FN + 'experiments_4_pedestrians' + str(fileID) + '.csv', delimiter=' ',
                                          dtype=float, usecols=(5,))

                self.clipLength = self.frameID.size
                self.robot = np.random.randint(1, int(max(self.pedID)))
                self.gtTrajIndex = np.argwhere(self.pedID == self.robot)
                if self.gtTrajIndex.size == 0:
                    distToGoal = 0
                else:
                    self.des = [self.pedPosX[self.gtTrajIndex[-1]], self.pedPosY[self.gtTrajIndex[-1]]]
                    distToGoal = np.sqrt((self.pedPosX[self.gtTrajIndex[0]] - self.des[0]) ** 2 + (
                        self.pedPosY[self.gtTrajIndex[0]] - self.des[1]) ** 2)

            maxVol = np.sqrt(max(self.pedVolX[self.gtTrajIndex]**2 + self.pedVolY[self.gtTrajIndex] ** 2))
            if maxVol < MAX_VT:
                self.SLOW_DOWN_RATE = 1
                self.COM_SLOW_DOWN_RATE = 1
            elif maxVol < 2 * MAX_VT:
                self.SLOW_DOWN_RATE = 2
                self.COM_SLOW_DOWN_RATE = 2
            else:
                self.SLOW_DOWN_RATE = 3
                self.COM_SLOW_DOWN_RATE = 3
            self.numOfInterpolation = int(self.FPS * self.SLOW_DOWN_RATE / T)
        else:
            if FN == './data/ewap_dataset_full/seq_eth/' or \
             FN == './data/ewap_dataset_full/seq_hotel/':
                self.FPS = 0.4
                minTrajSize = 10
            else:
                self.FPS = 0.1
                minTrajSize = 20
            self.frameID = np.loadtxt(FN + 'obsmat.csv', delimiter=' ', dtype=float, usecols=(0,))
            self.pedID = np.loadtxt(FN + 'obsmat.csv', delimiter=' ', dtype=float, usecols=(1,))
            self.pedPosX = np.loadtxt(FN + 'obsmat.csv', delimiter=' ', dtype=float, usecols=(2,))
            self.pedPosY = np.loadtxt(FN + 'obsmat.csv', delimiter=' ', dtype=float, usecols=(3,))
            self.pedVolX = np.loadtxt(FN + 'obsmat.csv', delimiter=' ', dtype=float, usecols=(4,))
            self.pedVolY = np.loadtxt(FN + 'obsmat.csv', delimiter=' ', dtype=float, usecols=(5,))

            self.clipLength = self.frameID.size

            self.robot = np.random.randint(1, int(max(self.pedID) + 1))

            self.gtTrajIndex = np.argwhere(self.pedID == self.robot)

            if self.gtTrajIndex.size == 0:
                distToGoal = 0
            else:
                self.des = [self.pedPosX[self.gtTrajIndex[-1]], self.pedPosY[self.gtTrajIndex[-1]]]
                distToGoal = np.sqrt((self.pedPosX[self.gtTrajIndex[0]] - self.des[0]) ** 2 + (
                    self.pedPosY[self.gtTrajIndex[0]] - self.des[1]) ** 2)

            while self.gtTrajIndex.size < minTrajSize or distToGoal < 5 or (float(distToGoal)/float(self.gtTrajIndex.size)/self.FPS) < 0.4:
                self.robot = np.random.randint(1, int(max(self.pedID)))
                self.gtTrajIndex = np.argwhere(self.pedID == self.robot)
                if self.gtTrajIndex.size == 0:
                    distToGoal = 0
                else:
                    self.des = [self.pedPosX[self.gtTrajIndex[-1]], self.pedPosY[self.gtTrajIndex[-1]]]
                    distToGoal = np.sqrt((self.pedPosX[self.gtTrajIndex[0]] - self.des[0]) ** 2 + (
                        self.pedPosY[self.gtTrajIndex[0]] - self.des[1]) ** 2)

            maxVol = np.sqrt(max(self.pedVolX[self.gtTrajIndex]**2 + self.pedVolY[self.gtTrajIndex] ** 2))
            if maxVol < MAX_VT:
                self.SLOW_DOWN_RATE = 1
                self.COM_SLOW_DOWN_RATE = 1
            elif maxVol < 2 * MAX_VT:
                self.SLOW_DOWN_RATE = 2
                self.COM_SLOW_DOWN_RATE = 2
            else:
                self.SLOW_DOWN_RATE = 3
                self.COM_SLOW_DOWN_RATE = 3
            self.numOfInterpolation = int(self.FPS * self.SLOW_DOWN_RATE / T)


        self.des = [self.pedPosX[self.gtTrajIndex[-1]], self.pedPosY[self.gtTrajIndex[-1]]]
        self.currentPosX = self.pedPosX[self.gtTrajIndex[0]]
        self.currentPosY = self.pedPosY[self.gtTrajIndex[0]]

        self.comFrame = 0
        currentX = np.copy(self.currentPosX)
        currentY = np.copy(self.currentPosY)
        self.rho = np.arctan2(self.des[1] - self.currentPosY,
                              self.des[0] - self.currentPosX)

        if self.compMode:
            self.comPosX = self.pedPosX[self.gtTrajIndex]
            self.comPosY = self.pedPosY[self.gtTrajIndex]
            self.comVolX = self.pedVolX[self.gtTrajIndex]
            self.comVolY = self.pedVolY[self.gtTrajIndex]

            while ((self.comPosX[self.comFrame] - currentX) ** 2 + \
                               (self.comPosY[self.comFrame]- currentY) ** 2) <= 0.6**2:
                self.comFrame += 1

            for i in xrange(0, NUMBER_OF_COM):
                dist2Com = np.sqrt(
                    (currentY - self.comPosY[self.comFrame]) ** 2 + (currentX - self.comPosX[self.comFrame]) ** 2)
                deltaRho = np.arctan2(self.comPosY[self.comFrame] - currentY,
                                      self.comPosX[self.comFrame] - currentX) - self.rho
                if deltaRho > np.pi:
                    deltaRho -= 2 * np.pi
                elif deltaRho < -np.pi:
                    deltaRho += 2 * np.pi
                self._state[4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = self._inject_obs_noise(
                    dist2Com), deltaRho
                self.trueState[4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = dist2Com, deltaRho
                self.comTrajX[i].append(self.comPosX[self.comFrame])
                self.comTrajY[i].append(self.comPosY[self.comFrame])
                self.dist2Com = dist2Com
                self.rho2Com = deltaRho
                if dist2Com >= GOAL_SWITCHING_BOUND[0]:
                    self._state[0:2] = dist2Com, deltaRho


        else:
            for i in xrange(0, NUMBER_OF_COM):
                dist2Com = DESIRED_COM_DISTANCE
                deltaRho = 0

                self.trueState[4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = dist2Com, deltaRho

                self._state[4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = self._inject_obs_noise(
                    dist2Com), deltaRho
                self.dist2Com = dist2Com
                self.rho2Com = deltaRho

        self.dist2Com = self.dist2Com / float(NUMBER_OF_COM)
        self.maxComDist = self.dist2Com

        currentFrame = self.frameID[self.gtTrajIndex[0]]
        startFrameID = np.argwhere(self.frameID == currentFrame)
        self.index = startFrameID[0]

        pedPosXBuffer,pedPosYBuffer = self.get_initial_state(currentFrame,FN)

        allDist2Ped = np.sqrt((self.currentPosX - pedPosXBuffer) ** 2 + (self.currentPosY - pedPosYBuffer) ** 2)
        index = np.argmin(allDist2Ped)
        deltaRho = np.arctan2(pedPosYBuffer[index] - self.currentPosY, pedPosXBuffer[index] - self.currentPosX) - self.rho
        if deltaRho > np.pi:
            deltaRho -= 2 * np.pi
        elif deltaRho < -np.pi:
            deltaRho += 2 * np.pi
        self.minPedRho = deltaRho

        for i in range(0,NUMBER_OF_PED):
            dist2Ped = allDist2Ped[i]

            self.trueState[4 + i * 2] = dist2Ped

            deltaRho = np.arctan2(pedPosYBuffer[i] - self.currentPosY, pedPosXBuffer[i] - self.currentPosX) - self.rho

            if deltaRho > np.pi:
                deltaRho -= 2 * np.pi
            elif deltaRho < -np.pi:
                deltaRho += 2 * np.pi

            self.trueState[4 + i * 2 + 1] = deltaRho
            if deltaRho>0.5*PED_DETECT_FOV or deltaRho<-0.5*PED_DETECT_FOV or dist2Ped > MAX_PED_DISTANCE:
                deltaRho = np.pi
                dist2Ped = MAX_PED_DISTANCE
            self._state[4 + i*2:4 + (i+1) * 2] = min(MAX_PED_DISTANCE,self._inject_obs_noise(dist2Ped)), deltaRho
            self.pedTrajX[i].append(pedPosXBuffer[i])
            self.pedTrajY[i].append(pedPosYBuffer[i])

        self.trajectoryX.append(currentX)
        self.trajectoryY.append(currentY)



        self._counter = 2
        self.comCounter = 2

        self.lastR = self.continuouRW([0,0])

        for i in xrange(0, STACK_FRAMES):
            self._stack.append(self._state)

        for i in xrange(0, STACK_FRAMES):
            self.trueStack.append(self.trueState)


        return self._stack, self.trueStack

    def step(self, action):
        action[0] = (action[0] + 1)/2 * 0.5 * MAX_VT + 0.5 * MAX_VT

        action[1] = action[1]*18*2*np.pi/360
        self.trackV.append(action[0])
        self.trackOmega.append(action[1])
        # update robot's Cartesian and angular position
        if abs(action[1]) >= 0.01:
            currentX = self.currentPosX - (action[0]/action[1])*(np.sin(self.rho)-np.sin(self.rho+action[1]*T))
            currentY = self.currentPosY + (action[0]/action[1])*(np.cos(self.rho)-np.cos(self.rho+action[1]*T))
            currentRho = self.rho + action[1] * T
            if currentRho > np.pi:
                currentRho -= 2*np.pi
            elif currentRho < -np.pi:
                currentRho += 2*np.pi

        else:
            currentX = self.currentPosX+ action[0]*np.cos(self.rho)*T
            currentY = self.currentPosY + action[0]*np.sin(self.rho)*T
            currentRho = self.rho
            if currentRho > np.pi:
                currentRho -= 2 * np.pi
            elif currentRho < -np.pi:
                currentRho += 2 * np.pi
        self.currentPosX = currentX
        self.currentPosY = currentY
        self.rho = currentRho
        self.DistToGoal = np.sqrt((self.currentPosX - self.des[0]) ** 2 + (self.currentPosY - self.des[1]) ** 2)
        self.RhoToGoal = np.arctan2(self.des[1] - self.currentPosY, self.des[0] - self.currentPosX)
        deltaRho = self.RhoToGoal - self.rho
        if deltaRho > np.pi:
            deltaRho -= 2 * np.pi
        elif deltaRho < -np.pi:
            deltaRho += 2 * np.pi


        self._state[0:4] = self.DistToGoal, deltaRho, action[0], action[1]
        self.trueState[0:4] = self.DistToGoal, deltaRho, action[0], action[1]

        self.trajectoryX.append(currentX)
        self.trajectoryY.append(currentY)

        pedPosXBuffer = BOUND_OF_SPACE * np.ones(NUMBER_OF_PED)
        pedPosYBuffer = BOUND_OF_SPACE * np.ones(NUMBER_OF_PED)
        if self.index<self.clipLength and self._counter > self.numOfInterpolation:
            self._counter = 1
            currentFrame = self.frameID[self.index]
            aIndex = self.index
            maxDist = (self.currentPosX - pedPosXBuffer[0]) ** 2 + (self.currentPosY - pedPosYBuffer[0]) ** 2
            distBuffer = maxDist * np.ones(NUMBER_OF_PED)
            newIDBuffer = []

            while self.frameID[aIndex] == currentFrame:
                whereID = np.argwhere(self.nearPedID == self.pedID[aIndex])
                if whereID:
                    if whereID.size > 1:
                        print "more than one ID found in buffer !"
                    else:
                        self.nearPedID[whereID[0][0]] = self.pedID[aIndex]
                        pedPosXBuffer[whereID[0][0]] = self.pedPosX[aIndex]
                        pedPosYBuffer[whereID[0][0]] = self.pedPosY[aIndex]
                        distBuffer[whereID[0][0]] = (self.currentPosX - self.pedPosX[aIndex]) ** 2 + (self.currentPosY -
                                                                                                    self.pedPosY[
                                                                                                        aIndex]) ** 2
                else:
                    np.append(newIDBuffer, aIndex)
                aIndex += 1
                if self.index >= self.clipLength:
                    break

            maxDistIndex = np.argmax(distBuffer)
            maxDist = distBuffer[maxDistIndex]
            for newID in newIDBuffer:
                aDist = (self.currentPosX - self.pedPosX[newID]) ** 2 + (self.currentPosY - self.pedPosY[newID]) ** 2
                if aDist < maxDist and self.pedID[newID]!=self.robot:
                    pedPosXBuffer[maxDistIndex] = self.pedPosX[newID]
                    pedPosYBuffer[maxDistIndex] = self.pedPosY[newID]
                    self.nearPedID[maxDistIndex] = self.pedID[newID]
                    distBuffer[maxDistIndex] = aDist
                    maxDistIndex = np.argmax(distBuffer)
                    maxDist = distBuffer[maxDistIndex]
            self.minPedDist = np.sqrt(min(distBuffer))
            self.index = aIndex
            self._counter += 1
        else:
            aIndex = self.index-1
            currentFrame = self.frameID[aIndex]
            maxDist = (self.currentPosX - pedPosXBuffer[0]) ** 2 + (self.currentPosY - pedPosYBuffer[0]) ** 2
            distBuffer = maxDist * np.ones(NUMBER_OF_PED)
            newIDBuffer = []

            while self.frameID[aIndex] == currentFrame:
                whereID = np.argwhere(self.nearPedID == self.pedID[aIndex])
                if whereID:
                    if whereID.size > 1:
                        print "more than one ID found in buffer !"
                    else:
                        self.nearPedID[whereID[0][0]] = self.pedID[aIndex]
                        aPosX = self.pedPosX[aIndex] + T*(self._counter-1)*self.pedVolX[aIndex]/self.SLOW_DOWN_RATE
                        aPosY = self.pedPosY[aIndex] + T*(self._counter-1)*self.pedVolY[aIndex]/self.SLOW_DOWN_RATE
                        pedPosXBuffer[whereID[0][0]] = aPosX
                        pedPosYBuffer[whereID[0][0]] = aPosY
                        distBuffer[whereID[0][0]] = (self.currentPosX-aPosX)**2+(self.currentPosY-aPosY) ** 2
                else:
                    np.append(newIDBuffer, aIndex)
                aIndex -= 1
                if self.index >= self.clipLength:
                    break

            maxDistIndex = np.argmax(distBuffer)
            maxDist = distBuffer[maxDistIndex]
            for newID in newIDBuffer:
                aPosX = self.pedPosX[newID] + T*(self._counter-1)*self.pedVolX[newID]/self.SLOW_DOWN_RATE
                aPosY = self.pedPosY[newID] + T*(self._counter-1)*self.pedVolY[newID]/self.SLOW_DOWN_RATE
                aDist = (self.currentPosX- aPosX) ** 2 + (self.currentPosY - aPosY) ** 2
                if aDist < maxDist and self.pedID[newID]!=self.robot:
                    pedPosXBuffer[maxDistIndex] = aPosX
                    pedPosYBuffer[maxDistIndex] = aPosY
                    distBuffer[maxDistIndex] = aDist
                    maxDistIndex = np.argmax(distBuffer)
                    maxDist = distBuffer[maxDistIndex]
                    self.nearPedID[maxDistIndex] = self.pedID[newID]

            self.minPedDist = np.sqrt(min(distBuffer))
            self._counter += 1

        allDist2Ped = np.sqrt((self.currentPosX - pedPosXBuffer) ** 2 + (self.currentPosY - pedPosYBuffer) ** 2)
        index = np.argmin(allDist2Ped)
        deltaRho = np.arctan2(pedPosYBuffer[index] - self.currentPosY,
                              pedPosXBuffer[index] - self.currentPosX) - self.rho
        if deltaRho > np.pi:
            deltaRho -= 2 * np.pi
        elif deltaRho < -np.pi:
            deltaRho += 2 * np.pi
        self.minPedRho = deltaRho

        for i in range(0, NUMBER_OF_PED):
            dist2Ped = np.sqrt((self.currentPosX - pedPosXBuffer[i]) ** 2 + (self.currentPosY - pedPosYBuffer[i]) ** 2)

            self.trueState[4 + i*2] = dist2Ped

            deltaRho = np.arctan2(pedPosYBuffer[i] - self.currentPosY, pedPosXBuffer[i] - self.currentPosX) - self.rho
            if deltaRho > np.pi:
                deltaRho -= 2 * np.pi
            elif deltaRho < -np.pi:
                deltaRho += 2 * np.pi

            self.trueState[4 + i*2+1]=deltaRho

            if deltaRho>0.5*PED_DETECT_FOV or deltaRho<-0.5*PED_DETECT_FOV or dist2Ped>MAX_PED_DISTANCE:
                deltaRho = np.pi
                dist2Ped = MAX_PED_DISTANCE

            self._state[4 + i*2:4 + (i+1) * 2] = min(MAX_PED_DISTANCE,self._inject_obs_noise(dist2Ped)), deltaRho

            self.pedTrajX[i].append(pedPosXBuffer[i])
            self.pedTrajY[i].append(pedPosYBuffer[i])


        collideCom = False
        lostCom = False
        self.dist2Com = 0.0
        if self.compMode:
            if self.comFrame < len(self.comPosX) - 1 and self.comCounter > int(self.FPS * self.COM_SLOW_DOWN_RATE / T):
                self.comFrame += 1
                for i in xrange(0, NUMBER_OF_COM):
                    dist2Com = np.sqrt(
                        (currentY - self.comPosY[self.comFrame]) ** 2 + (currentX - self.comPosX[self.comFrame]) ** 2)
                    deltaRho = np.arctan2(self.comPosY[self.comFrame] - currentY,
                                          self.comPosX[self.comFrame] - currentX) - self.rho
                    if deltaRho > np.pi:
                        deltaRho -= 2 * np.pi
                    elif deltaRho < -np.pi:
                        deltaRho += 2 * np.pi
                    self._state[
                    4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = self._inject_obs_noise(
                        dist2Com), deltaRho
                    self.trueState[
                    4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = dist2Com, deltaRho
                    self.comTrajX[i].append(self.comPosX[self.comFrame])
                    self.comTrajY[i].append(self.comPosY[self.comFrame])
                    if dist2Com < MIN_COM_DIST:
                        collideCom = True
                    if dist2Com >= GOAL_SWITCHING_BOUND[0]:
                        self._state[0:2] = dist2Com, deltaRho
                    self.comCounter = 2
                    self.dist2Com = dist2Com
                    self.rho2Com = deltaRho
            elif self.comFrame < len(self.comPosX) - 1 and self.comCounter <= int(self.FPS * self.COM_SLOW_DOWN_RATE / T):
                currentComPosX = self.comPosX[self.comFrame] + \
                                 T * (self.comCounter - 1) * self.comVolX[self.comFrame] / self.COM_SLOW_DOWN_RATE
                currentComPosY = self.comPosY[self.comFrame] + \
                                 T * (self.comCounter - 1) * self.comVolY[self.comFrame] / self.COM_SLOW_DOWN_RATE
                for i in xrange(0, NUMBER_OF_COM):
                    dist2Com = np.sqrt(
                        (currentY - currentComPosY) ** 2 + (currentX - currentComPosX) ** 2)
                    deltaRho = np.arctan2(currentComPosY - currentY,
                                          currentComPosX - currentX) - self.rho
                    if deltaRho > np.pi:
                        deltaRho -= 2 * np.pi
                    elif deltaRho < -np.pi:
                        deltaRho += 2 * np.pi
                    self._state[
                    4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = self._inject_obs_noise(
                        dist2Com), deltaRho
                    self.trueState[
                    4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = dist2Com, deltaRho
                    self.comTrajX[i].append(currentComPosX)
                    self.comTrajY[i].append(currentComPosY)
                    if dist2Com < MIN_COM_DIST:
                        collideCom = True
                    if dist2Com >= GOAL_SWITCHING_BOUND[0]:
                        self._state[0:2] = dist2Com, deltaRho
                    self.comCounter += 1
                    self.dist2Com = dist2Com
                    self.rho2Com = deltaRho
            else:
                currentComPosX = self.comPosX[-1]
                currentComPosY = self.comPosY[-1]
                for i in xrange(0, NUMBER_OF_COM):
                    dist2Com = np.sqrt(
                        (currentY - currentComPosY) ** 2 + (currentX - currentComPosX) ** 2)
                    deltaRho = np.arctan2(currentComPosY - currentY,
                                          currentComPosX - currentX) - self.rho
                    if deltaRho > np.pi:
                        deltaRho -= 2 * np.pi
                    elif deltaRho < -np.pi:
                        deltaRho += 2 * np.pi
                    self._state[
                    4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = self._inject_obs_noise(
                        dist2Com), deltaRho
                    self.trueState[
                    4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = dist2Com, deltaRho
                    self.comTrajX[i].append(currentComPosX)
                    self.comTrajY[i].append(currentComPosY)
                    if dist2Com < MIN_COM_DIST:
                        collideCom = True
                    if dist2Com >= GOAL_SWITCHING_BOUND[0]:
                        self._state[0:2] = dist2Com, deltaRho
                    self.dist2Com = dist2Com
                    self.rho2Com = deltaRho

            self.dist2Com = self.dist2Com / NUMBER_OF_COM
        else:
            for i in xrange(0, NUMBER_OF_COM):
                dist2Com = DESIRED_COM_DISTANCE
                deltaRho = self.RhoToGoal - self.rho
                if deltaRho > np.pi:
                    deltaRho -= 2 * np.pi
                elif deltaRho < -np.pi:
                    deltaRho += 2 * np.pi
                self._state[4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = self._inject_obs_noise(
                    dist2Com), deltaRho

                self.trueState[4 + 2 * NUMBER_OF_PED + 2 * i:4 + 2 * NUMBER_OF_PED + 2 * (i + 1)] = dist2Com, deltaRho

                if dist2Com < MIN_COM_DIST:
                    collideCom = True
                self.dist2Com = dist2Com

            self.dist2Com = self.dist2Com / NUMBER_OF_COM

        self.maxComDist = max(self.maxComDist,self.dist2Com)
        self.get_obstacles_dist()

        self._t += 1
        collidePed = False
        collideObs = False

        if self.minObsDist < MIN_OBS_DIST:
            collideObs = True

        if self.minPedDist < MIN_PED_DIST:
            collidePed = True

        if self.dist2Com > MAX_COM_DIST:
            lostCom = True

        reachTarget = self.DistToGoal < REACH_TARGET_BOUND

        done = (reachTarget or collidePed or collideObs or collideCom or lostCom or self._t>=399) and (self._t>=GOD_TIME)
        self.done = done

        self._stack.popleft()
        self._stack.append(self._state)

        self.trueStack.popleft()
        self.trueStack.append(self.trueState)

        next_observation = np.copy(self._stack)
        next_state = np.copy(self.trueStack)

        if done:
            if reachTarget:
                reward = 10000
                if self.compMode:
                    print self._t, "Reach Target with Com!"
                else:
                    print self._t, "Reach Target!"
            else:
                reward = -10000
                if collideObs:
                    print self._t, "Hit Obs!"
                elif collidePed:
                    print self._t, "Hit Ped!"
                elif collideCom:
                    print self._t, "Hit Com!"
                elif lostCom:
                    print self._t, "Lost Com!"
                else:
                    reward = -10000
                    print "Cannot end in 40s!"
        else:
            # reward = np.exp(-(1-self.DistToGoal/self.totalLength))
            # reward = min(self.minPedDist - MIN_PED_DIST, 2)
            # obsF = self.trueState[4 + 2 * NUMBER_OF_PED + 2 * NUMBER_OF_COM]
            # reward += min(obsF - MIN_OBS_DIST, 2)
            # reward += -min(abs(self.dist2Com - DESIRED_COM_DISTANCE), 3)
            # reward -= abs(action[1])
            # # reward = reward * 0.1
            reward = -10 * abs(action[1]*10/np.pi)
            # R = self.continuouRW(action)
            # reward -= (R - self.lastR)
            # self.lastR = R

        self.lastDistToGoal = self.DistToGoal

        if done:
            self.plotCount = self.plotCount % 10 + 1
            if self.plotCount % 5 == 0:
                self.show()
        if not np.isscalar(reward):
            reward = reward[0]
        return Step(observation=next_observation.flatten(), reward=reward, done=done,states=next_state.flatten())

    def render(self):
        pass

    def show(self):

        # print self.initialRho
        # print self.des
        print np.mean(self.trackV), np.mean(np.abs(self.trackOmega)), "MaxComDist: ", self.maxComDist
        plt.figure
        for i in range(0,NUMBER_OF_PED):
            plt.scatter(self.pedTrajX[i], self.pedTrajY[i], color='blue')
            plt.scatter(self.pedTrajX[i][-1], self.pedTrajY[i][-1], color='green', marker='v', s=20)
        if self.compMode:
            for i in range(0, NUMBER_OF_COM):
                plt.scatter(self.comTrajX[i], self.comTrajY[i], color='orange')
                plt.scatter(self.comTrajX[i][-1], self.comTrajY[i][-1], color='green', marker='v', s=20)
        plt.scatter(self.trajectoryX, self.trajectoryY)
        plt.scatter(self.trajectoryX[-1], self.trajectoryY[-1], color='green',marker='v',s=20)
        plt.scatter(self.obsCor[:, 0], self.obsCor[:, 1],color = 'green',s=2)
        plt.scatter(self.des[0],self.des[1], color = 'red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig('results.png')
        if self.DistToGoal < REACH_TARGET_BOUND:
            plt.savefig('results_suc.png')
        else:
            plt.savefig('results_fail.png')
        plt.clf()

    def _inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation.
        """
        noise = np.random.normal(0,obs*0.01/3.0)
        return obs + noise

    def get_initial_state(self,currentFrame,FN):
        self.totalLength = np.sqrt((self.currentPosY - self.des[1]) ** 2 + (self.currentPosX - self.des[0]) ** 2)
        self.DistToGoal = np.sqrt((self.currentPosX - self.des[0]) ** 2 + (self.currentPosY - self.des[1]) ** 2)
        self.lastDistToGoal = self.DistToGoal
        self.RhoToGoal = np.arctan2(self.des[1] - self.currentPosY, self.des[0] - self.currentPosX)
        self.initialRho = [self.rho, self.RhoToGoal]
        self._state[0:4] = np.copy(self.DistToGoal), 0, \
                           0, 0
        self.trueState[0:4] = np.copy(self.DistToGoal), 0, \
                           0, 0
        pedPosXBuffer = BOUND_OF_SPACE * np.ones(NUMBER_OF_PED)
        pedPosYBuffer = BOUND_OF_SPACE * np.ones(NUMBER_OF_PED)
        self.nearPedID = np.zeros(NUMBER_OF_PED)
        maxDistIndex = 0
        maxDist = (self.currentPosX - pedPosXBuffer[maxDistIndex]) ** 2 + (self.currentPosY - pedPosYBuffer[
            maxDistIndex]) ** 2
        distBuffer = maxDist * np.ones(NUMBER_OF_PED)

        while self.frameID[self.index] == currentFrame:
            aDist = (self.currentPosX - self.pedPosX[self.index]) ** 2 + (self.currentPosY - self.pedPosY[
                self.index]) ** 2
            if aDist < maxDist and self.pedID[self.index] != self.robot:
                self.nearPedID[maxDistIndex] = self.pedID[self.index]
                pedPosXBuffer[maxDistIndex] = self.pedPosX[self.index]
                pedPosYBuffer[maxDistIndex] = self.pedPosY[self.index]
                distBuffer[maxDistIndex] = aDist
                maxDistIndex = np.argmax(distBuffer)
                maxDist = distBuffer[maxDistIndex]
            self.index += 1
            if self.index >= self.clipLength:
                break

        self.minPedDist = np.sqrt(min(distBuffer))

        if os.path.isfile(FN + 'obscor.npy'):
            self.obsCor = np.load(FN + 'obscor.npy')
        else:
            self.obsCor = np.array([])
        self.get_obstacles_dist()

        return pedPosXBuffer,pedPosYBuffer

    def find_dist_and_rho(self, distArray, rhoArray, getMin):
        if distArray.size == 1:
            distArray = np.array([distArray])
        if getMin:
            index = np.argmin(distArray)
        else:
            index = np.argmax(distArray)
        return np.sqrt(distArray[index]), rhoArray[index]

    def get_obstacles_dist(self):
        currentPosX = self.currentPosX
        currentPosY = self.currentPosY

        obsF = MAX_OBS_DISTANCE
        hatObsF = obsF
        obsL = MAX_OBS_DISTANCE
        hatObsL = obsL
        rhoL = 0.5 * OBS_DETECT_FOV
        hatRhoL = rhoL
        obsLMax = MAX_OBS_DISTANCE
        hatObsLMax = obsLMax
        rhoLMax = min(0.5 * np.pi, 0.5 * OBS_DETECT_FOV)
        hatRhoLMax = rhoLMax
        obsR = MAX_OBS_DISTANCE
        hatObsR = obsR
        rhoR = -0.5 * OBS_DETECT_FOV
        hatRhoR = rhoR
        obsRMax = MAX_OBS_DISTANCE
        hatObsRMax = obsRMax
        rhoRMax = max(-0.5 * np.pi, -0.5 * OBS_DETECT_FOV)
        hatRhoRMax = rhoRMax

        if self.obsCor.size:
            allObsCorX = self.obsCor[:, 0]
            allObsCorY = self.obsCor[:, 1]
            allAllDist = (allObsCorX - currentPosX) ** 2 + (allObsCorY - currentPosY) ** 2
            index = allAllDist <= (MAX_OBS_DISTANCE ** 2)
            if np.any(index):
                allDist = allAllDist[index]
                obsCorX = allObsCorX[index]
                obsCorY = allObsCorY[index]
                allRho = np.arctan2(obsCorY - currentPosY, obsCorX - currentPosX) - self.rho
                for i in range(0, len(allRho)):
                    if allRho[i] > np.pi:
                        allRho[i] -= 2 * np.pi
                    elif allRho[i] < -np.pi:
                        allRho[i] += 2 * np.pi
                fIndex = np.argmin(abs(allRho))
                if abs(allRho[fIndex]) > 5. / 180. * np.pi:
                    obsF = MAX_OBS_DISTANCE
                    hatObsF = obsF
                else:
                    obsF = np.sqrt(allDist[fIndex])
                    hatObsF = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsF))

                allLeftIndex = allRho > 5. / 180. * np.pi
                if np.any(allLeftIndex):
                    tempDist = allDist[allLeftIndex]
                    tempRho = allRho[allLeftIndex]
                    if tempDist.size == 1:
                        tempDist = np.array([tempDist])
                        tempRho = np.array([tempRho])
                    obsL, rhoL = self.find_dist_and_rho(tempDist, tempRho, getMin=True)
                    inFOVIndex = tempRho <= 0.5 * OBS_DETECT_FOV
                    if rhoL <= 0.5 * OBS_DETECT_FOV:
                        hatRhoL = rhoL
                        hatObsL = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsL))
                    else:
                        if np.any(inFOVIndex):
                            hatObsL, hatRhoL = self.find_dist_and_rho(tempDist[inFOVIndex], tempRho[inFOVIndex],
                                                                      getMin=True)
                            hatObsL = min(MAX_OBS_DISTANCE, self._inject_obs_noise(hatObsL))
                        else:
                            hatRhoL = rhoL
                            hatObsL = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsL))

                    obsLMax, rhoLMax = self.find_dist_and_rho(tempDist, tempRho, getMin=False)
                    if rhoLMax <= 0.5 * OBS_DETECT_FOV:
                        hatRhoLMax = rhoLMax
                        hatObsLMax = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsLMax))
                    else:
                        if np.any(inFOVIndex):
                            hatObsLMax, hatRhoLMax = self.find_dist_and_rho(tempDist[inFOVIndex], tempRho[inFOVIndex],
                                                                            getMin=False)
                            hatObsLMax = min(MAX_OBS_DISTANCE, self._inject_obs_noise(hatObsLMax))
                        else:
                            hatRhoLMax = rhoLMax
                            hatObsLMax = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsLMax))

                allRightIndex = allRho < -5. / 180. * np.pi
                if np.any(allRightIndex):
                    tempDist = allDist[allRightIndex]
                    tempRho = allRho[allRightIndex]
                    if tempDist.size == 1:
                        tempDist = np.array([tempDist])
                        tempRho = np.array([tempRho])
                    obsR, rhoR = self.find_dist_and_rho(tempDist, tempRho, getMin=True)
                    inFOVIndex = tempRho >= -0.5 * OBS_DETECT_FOV
                    if rhoR >= -0.5 * OBS_DETECT_FOV:
                        hatRhoR = rhoR
                        hatObsR = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsR))
                    else:
                        if np.any(inFOVIndex):
                            hatObsR, hatRhoR = self.find_dist_and_rho(tempDist[inFOVIndex], tempRho[inFOVIndex],
                                                                      getMin=True)
                            hatObsR = min(MAX_OBS_DISTANCE, self._inject_obs_noise(hatObsR))
                        else:
                            hatRhoR = rhoR
                            hatObsR = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsR))

                    obsRMax, rhoRMax = self.find_dist_and_rho(tempDist, tempRho, getMin=False)
                    if rhoRMax >= -0.5 * OBS_DETECT_FOV:
                        hatRhoRMax = rhoRMax
                        hatObsRMax = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsRMax))
                    else:
                        if np.any(inFOVIndex):
                            hatObsRMax, hatRhoRMax = self.find_dist_and_rho(tempDist[inFOVIndex], tempRho[inFOVIndex],
                                                                            getMin=False)
                            hatObsRMax = min(MAX_OBS_DISTANCE, self._inject_obs_noise(hatObsRMax))
                        else:
                            hatRhoRMax = rhoRMax
                            hatObsRMax = min(MAX_OBS_DISTANCE, self._inject_obs_noise(obsRMax))

        self.minObsDist = min(obsF,obsL,obsR)
        self._state[
        4 + 2 * NUMBER_OF_PED + 2 * NUMBER_OF_COM:13 + 2 * NUMBER_OF_PED + 2 * NUMBER_OF_COM] = hatObsF, hatObsL, hatRhoL,\
                                                                                               hatObsR, hatRhoR,hatObsLMax,\
                                                                                               hatRhoLMax, hatObsRMax,hatRhoRMax
        self.trueState[
        4 + 2 * NUMBER_OF_PED + 2 * NUMBER_OF_COM:13 + 2 * NUMBER_OF_PED + 2 * NUMBER_OF_COM] = obsF, obsL, rhoL, obsR,\
                                                                                               rhoR, obsLMax,rhoLMax,obsRMax,rhoRMax


    def getMinDist(self,FN):
        if os.path.isfile(FN + 'obscor.npy'):
            self.obsCor = np.load(FN + 'obscor.npy')
            dist=np.square(self.obsCor[:,0]-self.pedPosX[self.gtTrajIndex[0]])+np.square(self.obsCor[:,1]-self.pedPosY[self.gtTrajIndex[0]])
            return np.sqrt(min(dist))
        else:
            return MAX_COM_DIST

    def continuouRW(self,action):
        reward = -100. / ((abs(min(self.minPedDist, MAX_PED_DISTANCE)-MIN_PED_DIST)) + 1e-8)
        return reward