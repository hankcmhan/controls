from operator import pos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from numpy.lib.twodim_base import eye
import control

g = 9.81 # gravity
class plantModel():
    def __init__(self, mass, theta) -> None:
        self.mass = mass # mass of the robot
        self.theta = theta # inclined angle of the ramp
    
    def processUpdate(self, state, dt, u):
        # state space. u: input
        A = np.array([[0, 1],
                      [0, 0]])
        B = np.array([[0], 
                      [u/self.mass - g*np.sin(self.theta)]])
        C = np.eye(2)
        newState = np.matmul((A*dt + np.eye(2)), state) + B*dt
        output = C.dot(newState)
        return output


class controllers():
    def __init__(self) -> None:
        self.prevTime = 0
        self.prevError = 0
        self.I = 0
        self.D = 0
        
    def onOff(self, error):
        if error < 0:
            u = 25
        else:
            u = 0
        return u

    def PID(self, error, currTime, params=[1, 0, 0]):
        Kp = params[0]
        Ki = params[1]
        Kd = params[2]
        diffTime = currTime - self.prevTime
        diffError = error - self.prevError
        self.I += error * diffTime
        if diffTime > 0:
            self.D = diffError/diffTime
        self.prevTime = currTime
        self.prevError = error
        return Kp*error + (Ki * self.I) + (Kd * self.D)

def main():
    # initialization
    tInit = 0
    tEnd = 100
    tStep = 500
    tFrame, dt = np.linspace(tInit, tEnd, num=tStep, retstep=True)
    mass = 10
    inclineAngle = np.pi/180*10
    setpoint = 50
    initState = np.array([[0], [0]])
    robot = plantModel(mass, inclineAngle)
    controller = controllers()

    # update process
    state = initState
    params = [5, 0.05, 10]
    rampPosData = [initState[0][0]]
    rampVelData = [initState[1][0]]
    uData = []
    for t in tFrame:
        currPos = rampPosData[-1]
        e = setpoint - currPos
        u = controller.PID(e, t, params)
        state = robot.processUpdate(state, dt, u)
        uData.append(u)
        rampPosData.append(state[0][0])
        rampVelData.append(state[1][0])

    xPos = [x*np.cos(inclineAngle) for x in rampPosData]
    yPos = [y*np.sin(inclineAngle) for y in rampPosData]
    # make a plot and animate the result
    fig1 = plt.figure(figsize=(8,6))
    ax1 = fig1.add_subplot(211)
    ax1.plot(tFrame, rampPosData[:len(tFrame)])
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('position(m)')
    ax2 = fig1.add_subplot(212)
    ax2.plot(tFrame, uData)
    ax2.set_xlabel('time(s)')
    ax2.set_ylabel('control(N)')

    # animation
    fig2 = plt.figure()
    ax3 = plt.axes()
    ax3.plot([0, 100], [0, 100*np.tan(inclineAngle)], 'k', linewidth = 3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Robot Lateral Control Simulation')
    plt.grid()

    robot = mpatches.Rectangle((0,0), 8, 4, facecolor='b', fill=True)
    ax3.add_patch(robot)
    
    def init():
        robot.set_xy([xPos[0], yPos[0]])
        return robot,
    
    def animate(i):
        robot.set_xy([xPos[i], yPos[i]])
        return robot,

    ani = animation.FuncAnimation(fig2, animate, frames=len(tFrame), blit=True, interval=20)
    plt.show()

if __name__ == '__main__':
    main()