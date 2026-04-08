import tkinter as tk
import random
import math
import numpy as np


class Brain():

    def __init__(self, botp):
        self.bot = botp
        self.turningCount = 0
        self.movingCount = random.randrange(50, 100)
        self.currentlyTurning = False
        # ========== 避障相关变量 ==========
        self.avoiding_obstacle = False  # 是否正在避障
        self.obstacle_turn_direction = 1  # 转弯方向（1=左转，-1=右转）
        self.obstacle_turn_steps = 0  # 还需要转多少步
        # self

    def thinkAndAct(self, lightL, lightR, chargerL, chargerR, x, y, sl, sr, battery):
        newX = None
        newY = None

        # ========== 1. 避让墙壁（最高优先级） ==========
        WALL_DISTANCE = 50
        CANVAS_SIZE = 1000

        touching_left = (x < WALL_DISTANCE)
        touching_right = (x > CANVAS_SIZE - WALL_DISTANCE)
        touching_top = (y < WALL_DISTANCE)
        touching_bottom = (y > CANVAS_SIZE - WALL_DISTANCE)

        if touching_left or touching_right or touching_top or touching_bottom:
            # 重置避障状态
            self.avoiding_obstacle = False
            self.obstacle_turn_steps = 0

            if touching_left:
                speedLeft = 8.0
                speedRight = -8.0
            elif touching_right:
                speedLeft = -8.0
                speedRight = 8.0
            elif touching_top:
                speedLeft = 8.0
                speedRight = -8.0
            else:  # touching_bottom
                speedLeft = -8.0
                speedRight = 8.0

            if x < WALL_DISTANCE:
                newX = WALL_DISTANCE
            if x > CANVAS_SIZE - WALL_DISTANCE:
                newX = CANVAS_SIZE - WALL_DISTANCE
            if y < WALL_DISTANCE:
                newY = WALL_DISTANCE
            if y > CANVAS_SIZE - WALL_DISTANCE:
                newY = CANVAS_SIZE - WALL_DISTANCE

            return speedLeft, speedRight, newX, newY

        # ========== 2. 避让障碍物（第二优先级） ==========
        OBSTACLE_THRESHOLD = 150  # 信号强度阈值（降低一点，更敏感）

        # 检测是否靠近障碍物
        near_obstacle = (chargerL + chargerR > OBSTACLE_THRESHOLD)

        if near_obstacle:
            # 如果还没开始避障，初始化避障状态
            if not self.avoiding_obstacle:
                self.avoiding_obstacle = True
                # 根据信号强度决定转弯方向
                if chargerR > chargerL:
                    # 右边信号强，障碍物在右边，向左转
                    self.obstacle_turn_direction = 1  # 左转
                else:
                    # 左边信号强，障碍物在左边，向右转
                    self.obstacle_turn_direction = -1  # 右转
                # 设置转弯步数（30-50步，确保离开障碍物）
                self.obstacle_turn_steps = random.randrange(30, 50)

            # 执行避障转弯（连续转弯多步）
            if self.obstacle_turn_direction == 1:  # 左转
                speedLeft = -6.0
                speedRight = 6.0
            else:  # 右转
                speedLeft = 6.0
                speedRight = -6.0

            # 减少剩余步数
            self.obstacle_turn_steps -= 1

            # 转弯完成，退出避障模式
            if self.obstacle_turn_steps <= 0:
                self.avoiding_obstacle = False

            return speedLeft, speedRight, newX, newY

        # ========== 没有靠近障碍物时，重置避障状态 ==========
        self.avoiding_obstacle = False

        # ========== 3. 原有代码：随机游走 ==========
        if self.currentlyTurning == True:
            speedLeft = -2.0
            speedRight = 2.0
            self.turningCount -= 1
        else:
            speedLeft = 5.0
            speedRight = 5.0
            self.movingCount -= 1
        if self.movingCount == 0 and not self.currentlyTurning:
            self.turningCount = random.randrange(20, 40)
            self.currentlyTurning = True
        if self.turningCount == 0 and self.currentlyTurning:
            self.movingCount = random.randrange(50, 100)
            self.currentlyTurning = False

        # ========== 4. 低电量找充电桩 ==========
        if battery < 600:
            if chargerR > chargerL:
                speedLeft = 2.0
                speedRight = -2.0
            elif chargerR < chargerL:
                speedLeft = -2.0
                speedRight = 2.0
            if abs(chargerR - chargerL) < chargerL * 0.1:
                speedLeft = 5.0
                speedRight = 5.0

        # ========== 5. 靠近充电桩时充电 ==========
        if chargerL + chargerR > 200 and battery < 1000:
            speedLeft = 0.0
            speedRight = 0.0

        return speedLeft, speedRight, newX, newY
class Bot():

    def __init__(self,namep):
        self.name = namep
        self.x = random.randint(100,900)
        self.y = random.randint(100,900)
        self.theta = random.uniform(0.0,2.0*math.pi)
        #self.theta = 0
        self.ll = 60 #axle width
        self.sl = 0.0
        self.sr = 0.0
        self.battery = 1000

    def thinkAndAct(self, agents, passiveObjects):
        lightL, lightR = self.senseLight(passiveObjects)
        chargerL, chargerR = self.senseChargers(passiveObjects)
        self.sl, self.sr, xx, yy = self.brain.thinkAndAct\
            (lightL, lightR, chargerL, chargerR, self.x, self.y, self.sl, self.sr, self.battery)
        if xx != None:
            self.x = xx
        if yy != None:
            self.y = yy
        
    def setBrain(self,brainp):
        self.brain = brainp

    #returns the output from polling the light sensors
    def senseLight(self, passiveObjects):
        lightL = 0.0
        lightR = 0.0
        for pp in passiveObjects:
            if isinstance(pp,Lamp):
                lx,ly = pp.getLocation()
                distanceL = math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                                       (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )
                distanceR = math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                                       (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )
                lightL += 200000/(distanceL*distanceL)
                lightR += 200000/(distanceR*distanceR)
        return lightL, lightR

    #returns sensors values that detect chargers
    def senseChargers(self, passiveObjects):
        chargerL = 0.0
        chargerR = 0.0
        for pp in passiveObjects:
            # 检测充电桩和障碍物（都需要避开）
            if isinstance(pp, Charger) or isinstance(pp, Obstacle):
                lx, ly = pp.getLocation()
                distanceL = math.sqrt((lx - self.sensorPositions[0]) * (lx - self.sensorPositions[0]) +
                                      (ly - self.sensorPositions[1]) * (ly - self.sensorPositions[1]))
                distanceR = math.sqrt((lx - self.sensorPositions[2]) * (lx - self.sensorPositions[2]) +
                                      (ly - self.sensorPositions[3]) * (ly - self.sensorPositions[3]))
                chargerL += 200000 / (distanceL * distanceL)
                chargerR += 200000 / (distanceR * distanceR)
        return chargerL, chargerR

    def distanceTo(self,obj):
        xx,yy = obj.getLocation()
        return math.sqrt( math.pow(self.x-xx,2) + math.pow(self.y-yy,2) )

    # what happens at each timestep
    def update(self,canvas,passiveObjects,dt):
        # for now, the only thing that changes is that the robot moves
        #   (using the current settings of self.sl and self.sr)
        self.battery -= 1
        for rr in passiveObjects:
            if isinstance(rr,Charger) and self.distanceTo(rr)<80:
                self.battery += 10
        if self.battery<=0:
            self.battery = 0
        self.move(canvas,dt)

    # draws the robot at its current position
    def draw(self,canvas):
        points = [ (self.x + 30*math.sin(self.theta)) - 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - 30*math.cos(self.theta)) - 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - 30*math.sin(self.theta)) - 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + 30*math.cos(self.theta)) - 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - 30*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + 30*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x + 30*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - 30*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta)  \
                ]
        canvas.create_polygon(points, fill="blue", tags=self.name)

        self.sensorPositions = [ (self.x + 20*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y - 20*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta), \
                                 (self.x - 20*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y + 20*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta)  \
                            ]
    
        centre1PosX = self.x 
        centre1PosY = self.y
        canvas.create_oval(centre1PosX-16,centre1PosY-16,\
                           centre1PosX+16,centre1PosY+16,\
                           fill="gold",tags=self.name)
        canvas.create_text(self.x,self.y,text=str(self.battery),tags=self.name)

        wheel1PosX = self.x - 30*math.sin(self.theta)
        wheel1PosY = self.y + 30*math.cos(self.theta)
        canvas.create_oval(wheel1PosX-3,wheel1PosY-3,\
                                         wheel1PosX+3,wheel1PosY+3,\
                                         fill="red",tags=self.name)

        wheel2PosX = self.x + 30*math.sin(self.theta)
        wheel2PosY = self.y - 30*math.cos(self.theta)
        canvas.create_oval(wheel2PosX-3,wheel2PosY-3,\
                                         wheel2PosX+3,wheel2PosY+3,\
                                         fill="green",tags=self.name)

        sensor1PosX = self.sensorPositions[0]
        sensor1PosY = self.sensorPositions[1]
        sensor2PosX = self.sensorPositions[2]
        sensor2PosY = self.sensorPositions[3]
        canvas.create_oval(sensor1PosX-3,sensor1PosY-3, \
                           sensor1PosX+3,sensor1PosY+3, \
                           fill="yellow",tags=self.name)
        canvas.create_oval(sensor2PosX-3,sensor2PosY-3, \
                           sensor2PosX+3,sensor2PosY+3, \
                           fill="yellow",tags=self.name)

    # handles the physics of the movement
    # cf. Dudek and Jenkin, Computational Principles of Mobile Robotics
    def move(self,canvas,dt):
        print(self.battery)
        if self.battery==0:
            self.sl = 0
            self.sl = 0
        if self.sl==self.sr:
            R = 0
        else:
            R = (self.ll/2.0)*((self.sr+self.sl)/(self.sl-self.sr))
        omega = (self.sl-self.sr)/self.ll
        ICCx = self.x-R*math.sin(self.theta) #instantaneous centre of curvature
        ICCy = self.y+R*math.cos(self.theta)
        m = np.matrix( [ [math.cos(omega*dt), -math.sin(omega*dt), 0], \
                        [math.sin(omega*dt), math.cos(omega*dt), 0],  \
                        [0,0,1] ] )
        v1 = np.matrix([[self.x-ICCx],[self.y-ICCy],[self.theta]])
        v2 = np.matrix([[ICCx],[ICCy],[omega*dt]])
        newv = np.add(np.dot(m,v1),v2)
        newX = newv.item(0)
        newY = newv.item(1)
        newTheta = newv.item(2)
        newTheta = newTheta%(2.0*math.pi) #make sure angle doesn't go outside [0.0,2*pi)
        self.x = newX
        self.y = newY
        self.theta = newTheta        
        if self.sl==self.sr: # straight line movement
            self.x += self.sr*math.cos(self.theta) #sr wlog
            self.y += self.sr*math.sin(self.theta)
        canvas.delete(self.name)
        self.draw(canvas)

    def collectDirt(self, canvas, passiveObjects):
        toDelete = []
        for idx,rr in enumerate(passiveObjects):
            if isinstance(rr,Dirt):
                if self.distanceTo(rr)<30:
                    canvas.delete(rr.name)
                    toDelete.append(idx)
        for ii in sorted(toDelete,reverse=True):
            del passiveObjects[ii]
        return passiveObjects
        

class Lamp():
    def __init__(self,namep):
        self.centreX = random.randint(100,900)
        self.centreY = random.randint(100,900)
        self.name = namep
        
    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-10,self.centreY-10, \
                                  self.centreX+10,self.centreY+10, \
                                  fill="yellow",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY

class Charger():
    def __init__(self,namep):
        self.centreX = random.randint(100,900)
        self.centreY = random.randint(100,900)
        self.name = namep
        
    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-10,self.centreY-10, \
                                  self.centreX+10,self.centreY+10, \
                                  fill="red",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY

class WiFiHub:
    def __init__(self,namep,xp,yp):
        self.centreX = xp
        self.centreY = yp
        self.name = namep
        
    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-10,self.centreY-10, \
                                  self.centreX+10,self.centreY+10, \
                                  fill="purple",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY

class Dirt:
    def __init__(self,namep):
        self.centreX = random.randint(100,900)
        self.centreY = random.randint(100,900)
        self.name = namep

    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-1,self.centreY-1, \
                                  self.centreX+1,self.centreY+1, \
                                  fill="grey",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


class Obstacle:
    """障碍物类 - 机器人需要避让的物体（可被充电传感器检测）"""

    def __init__(self, namep, xp=None, yp=None):
        if xp is None:
            self.centreX = random.randint(50, 950)
        else:
            self.centreX = xp
        if yp is None:
            self.centreY = random.randint(50, 950)
        else:
            self.centreY = yp
        self.name = namep
        self.radius = 15

    def draw(self, canvas):
        body = canvas.create_oval(self.centreX - self.radius,
                                  self.centreY - self.radius,
                                  self.centreX + self.radius,
                                  self.centreY + self.radius,
                                  fill="brown", tags=self.name)
        canvas.create_line(self.centreX - 8, self.centreY - 8,
                           self.centreX + 8, self.centreY + 8,
                           fill="white", width=2, tags=self.name)
        canvas.create_line(self.centreX - 8, self.centreY + 8,
                           self.centreX + 8, self.centreY - 8,
                           fill="white", width=2, tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY

    def getRadius(self):
        return self.radius


def initialise(window):
    window.resizable(False,False)
    canvas = tk.Canvas(window,width=1000,height=1000)
    canvas.pack()
    return canvas

def buttonClicked(x,y,agents):
    for rr in agents:
        if isinstance(rr,Bot):
            rr.x = x
            rr.y = y


def createObjects(canvas, noOfBots=1, noOfLights=2, amountOfDirt=300, noOfObstacles=5):
    agents = []
    passiveObjects = []

    # 创建机器人
    for i in range(0, noOfBots):
        bot = Bot("Bot" + str(i))
        brain = Brain(bot)
        bot.setBrain(brain)
        agents.append(bot)
        bot.draw(canvas)

    # 创建光源
    for i in range(0, noOfLights):
        lamp = Lamp("Lamp" + str(i))
        passiveObjects.append(lamp)
        lamp.draw(canvas)

    # 创建充电桩
    charger = Charger("Charger")
    passiveObjects.append(charger)
    charger.draw(canvas)

    # 创建WiFi热点
    hub1 = WiFiHub("Hub1", 950, 50)
    passiveObjects.append(hub1)
    hub1.draw(canvas)
    hub2 = WiFiHub("Hub2", 50, 500)
    passiveObjects.append(hub2)
    hub2.draw(canvas)

    # ========== 新增：创建随机障碍物 ==========
    for i in range(0, noOfObstacles):
        obstacle = Obstacle("Obstacle" + str(i))
        passiveObjects.append(obstacle)
        obstacle.draw(canvas)

    # 创建灰尘
    for i in range(0, amountOfDirt):
        dirt = Dirt("Dirt" + str(i))
        passiveObjects.append(dirt)
        dirt.draw(canvas)

    canvas.bind("<Button-1>", lambda event: buttonClicked(event.x, event.y, agents))
    return agents, passiveObjects

def moveIt(canvas,agents,passiveObjects):
    for rr in agents:
        rr.thinkAndAct(agents,passiveObjects)
        rr.update(canvas,passiveObjects,1.0)
        passiveObjects = rr.collectDirt(canvas,passiveObjects)
    canvas.after(50,moveIt,canvas,agents,passiveObjects)

def main():
    window = tk.Tk()
    canvas = initialise(window)
    agents, passiveObjects = createObjects(canvas,noOfBots=1,noOfLights=0,amountOfDirt=300)
    moveIt(canvas,agents,passiveObjects)
    window.mainloop()

# main()
