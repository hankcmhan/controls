import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

g = 9.81 # gravity
class fallingBall():
    def __init__(self) -> None:
        self.stopped = False
        self.lastState = None
    
    def stop(self):
        self.stopped = True

    def restart(self):
        self.stopped = False
        self.lastState = None

    # the governing equation of a free falling ball. Direction: y.
    def motionModel(self, state, dt):
        if self.stopped: return self.lastState
        A = np.array([[0, 1],
                      [0, 0]])
        B = np.array([[0],
                      [-g]])
        newState = np.matmul((A*dt + np.eye(2)), state) + B*dt # Euler method
        self.lastState = newState
        return newState


class plantModel():
    def __init__(self, mass, theta) -> None:
        self.mass = mass # mass of the robot
        self.theta = theta # inclined angle of the ramp
        self.stopped = False
        self.lastState = None
    
    def stop(self):
        self.stopped = True
    
    def restart(self):
        self.stopped = False
        self.lastState = None
    
    def processUpdate(self, state, dt, u):
        if self.stopped == True: 
            self.lastState[1] = 0
            return self.lastState
        # state space. u: input
        # coordinate s: on the ramp
        A = np.array([[0, 1],
                      [0, 0]])
        B = np.array([[0], 
                      [u/self.mass - g*np.sin(self.theta)]])
        C = np.eye(2) # assume full state output. 
        newState = np.matmul((A*dt + np.eye(2)), state) + B*dt # Euler method
        output = C.dot(newState)
        self.lastState = output
        return output


class controllers():
    def __init__(self) -> None:
        self.prevTime = 0
        self.prevError = 0
        self.integralError = 0
        self.differentialError = 0
        
    def onOff(self, error):
        return 0.6*error if error > 0 else 0

    def PID(self, error, currTime, params=[1, 0, 0]):
        Kp = params[0]
        Ki = params[1]
        Kd = params[2]
        diffTime = currTime - self.prevTime
        diffError = error - self.prevError
        self.integralError += error * diffTime
        if diffTime > 0:
            self.differentialError = diffError/diffTime
        self.prevTime = currTime
        self.prevError = error
        return Kp*error + (Ki * self.integralError) + (Kd * self.differentialError)

def main():
    # system configuration
    robotMass = 5
    robotWidth = 4
    robotHeight = 3
    ballRadius = 1
    rampAngle = np.pi/180*5
    xRange = [0, 50] # the range in x where the ball is located
    yRange = [25, 50] # the range in y where the ball is located
    tInit = 0
    tEnd = 3
    tStep = 200
    pidParams = [25, 0.15, 15] # [P, I, D]
    robot = plantModel(robotMass, rampAngle)
    ball = fallingBall()
    controller = controllers()

    # initialization
    numOfBalls = 5
    tFrame, dt = np.linspace(tInit, tEnd, num=tStep, retstep=True)
    robotInitState = np.array([[0], [0]]) # the s coordiante. state: [[position], [velocity]]
    # data storages
    control = np.zeros((numOfBalls, len(tFrame)))
    robotPos = np.zeros((numOfBalls, len(tFrame)))
    robotVel = np.zeros((numOfBalls, len(tFrame)))
    robotPos_x = np.zeros((numOfBalls, len(tFrame)))
    robotPos_y = np.zeros((numOfBalls, len(tFrame)))
    ballPos_x = np.zeros((numOfBalls, len(tFrame)))
    ballPos_y = np.zeros((numOfBalls, len(tFrame)))
    
    # update process
    for i in range(numOfBalls):
        robotPos[i][0] = robotInitState[0][0]
        robotVel[i][0] = robotInitState[1][0]
        robotPos_x[i][0] = robotPos[i][0]*np.cos(rampAngle)
        robotPos_y[i][0] = robotPos[i][0]*np.sin(rampAngle)
        ballLocation = (np.random.randint(xRange[0], xRange[1]), np.random.randint(yRange[0], yRange[1]))
        ballPos_x[i][0] = ballLocation[0]
        ballPos_y[i][0] = ballLocation[1]
        ballYInitState = np.array([[ballLocation[1]], [-10]]) # the xy coordinate
        print('ball location: ({}, {})'.format(ballLocation[0], ballLocation[1]))

        # control the robot to catch the ball[i]
        robotState = robotInitState
        ballYState = ballYInitState
        for j in range(1, len(tFrame)):
            t = tFrame[j]
            # update the robot
            robotCurrPos = robotPos[i][j-1]
            setpoint = np.sqrt(ballPos_x[i][j-1]**2 + (ballPos_x[i][j-1]*np.tan(rampAngle))**2)
            error = setpoint - robotCurrPos
            u = controller.PID(error, t, pidParams)
            robotState = robot.processUpdate(robotState, dt, u)
            control[i][j] = u
            robotPos[i][j] = robotState[0][0]
            robotVel[i][j] = robotState[1][0]
            robotPos_x[i][j] = robotPos[i][j]*np.cos(rampAngle)
            robotPos_y[i][j] = robotPos[i][j]*np.sin(rampAngle)

            # update the ball
            ballYState = ball.motionModel(ballYState, dt)
            ballPos_x[i][j] = ballLocation[0]
            ballPos_y[i][j] = ballYState[0][0]

            # collision condition
            stopCond_x = [robotState[0][0]*np.cos(rampAngle)-robotWidth//2, robotState[0][0]*np.cos(rampAngle)+robotWidth//2]
            stopCond_y = [robotState[0][0]*np.sin(rampAngle), robotState[0][0]*np.sin(rampAngle)+robotHeight]
            if stopCond_x[0] < ballPos_x[i][j] < stopCond_x[1] and stopCond_y[0] < ballPos_y[i][j] < stopCond_y[1]:
                robot.stop()
                ball.stop()
                u = 0
        print('robotState: ({}, {})'.format(robotState[0], robotState[1]))
        robotInitState = robotState
        robot.restart()
        ball.restart()
    # make a plot and animate the result
    # animation
    fig2 = plt.figure()
    ax3 = plt.axes()
    ax3.plot([0, xRange[1]], [0, xRange[1]*np.tan(rampAngle)], 'k', linewidth = 3)
    plt.xlim(0, xRange[1])
    plt.ylim(0, yRange[1])
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.title('Ball Catching Control Simulation')
    plt.grid()

    robot = mpatches.Rectangle((0,0), robotWidth, robotHeight, facecolor='b', fill=True)
    ball = mpatches.Circle((0,0), ballRadius, facecolor='r', fill=True)
    ax3.add_patch(robot)
    ax3.add_patch(ball)
    
    # data transformation
    robotX = [robotPos_x[i][j] for i in range(numOfBalls) for j in range(len(tFrame))]
    robotY = [robotPos_y[i][j] for i in range(numOfBalls) for j in range(len(tFrame))]
    ballX = [ballPos_x[i][j] for i in range(numOfBalls) for j in range(len(tFrame))]
    ballY = [ballPos_y[i][j] for i in range(numOfBalls) for j in range(len(tFrame))]

    print(robotX)
    def init():
        robot.set_xy([robotX[0]-robotWidth//2, robotY[0]])
        ball.center = ballX[0], ballY[0]
        return robot, ball
    
    def animate(i):
        robot.set_xy([robotX[i]-robotWidth//2, robotY[i]])
        ball.center = ballX[i], ballY[i]
        return robot, ball

    ani = animation.FuncAnimation(fig2, animate, frames=numOfBalls*len(tFrame), blit=True, interval=3, repeat=False)
    # ani.save('myAnimation.gif', writer='imagemagick', fps=30)
    plt.show()

if __name__ == '__main__':
    main()