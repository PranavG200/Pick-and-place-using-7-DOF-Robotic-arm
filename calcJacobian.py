import numpy as np
try:
    from lib.calculateFK import FK
except:
    from calculateFK import FK




import numpy as np
from lib.calculateFK import FK
from numpy import pi, sin, cos

def calcJacobian(q):
    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    t1,t2,t3,t4,t5,t6,t7 = q
    t7 = t7-pi/4
    s1 = sin(t1)
    c1 = cos(t1)
    s2 = sin(t2)
    c2 = cos(t2)
    s3 = sin(t3)
    c3 = cos(t3)
    s4 = sin(t4)
    c4 = cos(t4)
    s5 = sin(t5)
    c5 = cos(t5)
    s6 = sin(t6)
    c6 = cos(t6)
    # s7 = sin(t7)
    # c7 = cos(t7)

    J = np.array([[0.21*(((-s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*c5 + (s1*s3*c2 - c1*c3)*s5)*s6 + 0.088*(((-s1*c2*c3 - s3*c1)*c4 - s1*s2*s4)*c5 + (s1*s3*c2 - c1*c3)*s5)*c6 + 0.088*(-(-s1*c2*c3 - s3*c1)*s4 - s1*s2*c4)*s6 - 0.21*(-(-s1*c2*c3 - s3*c1)*s4 - s1*s2*c4)*c6 - 0.384*(-s1*c2*c3 - s3*c1)*s4 - 0.0825*(-s1*c2*c3 - s3*c1)*c4 + 0.0825*s1*s2*s4 - 0.384*s1*s2*c4 - 0.316*s1*s2 - 0.0825*s1*c2*c3 - 0.0825*s3*c1, 0.21*((-s2*c1*c3*c4 + s4*c1*c2)*c5 + s2*s3*s5*c1)*s6 + 0.088*((-s2*c1*c3*c4 + s4*c1*c2)*c5 + s2*s3*s5*c1)*c6 + 0.088*(s2*s4*c1*c3 + c1*c2*c4)*s6 - 0.21*(s2*s4*c1*c3 + c1*c2*c4)*c6 + 0.384*s2*s4*c1*c3 + 0.0825*s2*c1*c3*c4 - 0.0825*s2*c1*c3 - 0.0825*s4*c1*c2 + 0.384*c1*c2*c4 + 0.316*c1*c2, 0.21*((s1*s3 - c1*c2*c3)*s5 + (-s1*c3 - s3*c1*c2)*c4*c5)*s6 + 0.088*((s1*s3 - c1*c2*c3)*s5 + (-s1*c3 - s3*c1*c2)*c4*c5)*c6 - 0.088*(-s1*c3 - s3*c1*c2)*s4*s6 + 0.21*(-s1*c3 - s3*c1*c2)*s4*c6 - 0.384*(-s1*c3 - s3*c1*c2)*s4 - 0.0825*(-s1*c3 - s3*c1*c2)*c4 - 0.0825*s1*c3 - 0.0825*s3*c1*c2, 0.21*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*s6*c5 + 0.088*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*c5*c6 + 0.088*(-(-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*s6 - 0.21*(-(-s1*s3 + c1*c2*c3)*c4 - s2*s4*c1)*c6 + 0.0825*(-s1*s3 + c1*c2*c3)*s4 - 0.384*(-s1*s3 + c1*c2*c3)*c4 - 0.384*s2*s4*c1 - 0.0825*s2*c1*c4, 0.21*(-((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*s6 + 0.088*(-((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*s5 + (-s1*c3 - s3*c1*c2)*c5)*c6, -0.088*(((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 + 0.21*(((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*c6 + 0.21*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*s6 + 0.088*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*c6, 0], [0.21*(((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 + 0.088*(((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*c6 + 0.088*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*s6 - 0.21*(-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*c6 - 0.384*(-s1*s3 + c1*c2*c3)*s4 - 0.0825*(-s1*s3 + c1*c2*c3)*c4 - 0.0825*s1*s3 - 0.0825*s2*s4*c1 + 0.384*s2*c1*c4 + 0.316*s2*c1 + 0.0825*c1*c2*c3, 0.21*((-s1*s2*c3*c4 + s1*s4*c2)*c5 + s1*s2*s3*s5)*s6 + 0.088*((-s1*s2*c3*c4 + s1*s4*c2)*c5 + s1*s2*s3*s5)*c6 + 0.088*(s1*s2*s4*c3 + s1*c2*c4)*s6 - 0.21*(s1*s2*s4*c3 + s1*c2*c4)*c6 + 0.384*s1*s2*s4*c3 + 0.0825*s1*s2*c3*c4 - 0.0825*s1*s2*c3 - 0.0825*s1*s4*c2 + 0.384*s1*c2*c4 + 0.316*s1*c2, 0.21*((-s1*s3*c2 + c1*c3)*c4*c5 + (-s1*c2*c3 - s3*c1)*s5)*s6 + 0.088*((-s1*s3*c2 + c1*c3)*c4*c5 + (-s1*c2*c3 - s3*c1)*s5)*c6 - 0.088*(-s1*s3*c2 + c1*c3)*s4*s6 + 0.21*(-s1*s3*c2 + c1*c3)*s4*c6 - 0.384*(-s1*s3*c2 + c1*c3)*s4 - 0.0825*(-s1*s3*c2 + c1*c3)*c4 - 0.0825*s1*s3*c2 + 0.0825*c1*c3, 0.21*(-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*s6*c5 + 0.088*(-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*c5*c6 + 0.088*(-(s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*s6 - 0.21*(-(s1*c2*c3 + s3*c1)*c4 - s1*s2*s4)*c6 + 0.0825*(s1*c2*c3 + s3*c1)*s4 - 0.384*(s1*c2*c3 + s3*c1)*c4 - 0.384*s1*s2*s4 - 0.0825*s1*s2*c4, 0.21*(-((s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*s6 + 0.088*(-((s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 + (-s1*s3*c2 + c1*c3)*c5)*c6, -0.088*(((s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 + 0.21*(((s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*c6 + 0.21*(-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*s6 + 0.088*(-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*c6, 0], [0, 0.21*((-s2*s4 - c2*c3*c4)*c5 + s3*s5*c2)*s6 + 0.088*((-s2*s4 - c2*c3*c4)*c5 + s3*s5*c2)*c6 + 0.088*(-s2*c4 + s4*c2*c3)*s6 - 0.21*(-s2*c4 + s4*c2*c3)*c6 + 0.0825*s2*s4 - 0.384*s2*c4 - 0.316*s2 + 0.384*s4*c2*c3 + 0.0825*c2*c3*c4 - 0.0825*c2*c3, 0.21*(s2*s3*c4*c5 + s2*s5*c3)*s6 + 0.088*(s2*s3*c4*c5 + s2*s5*c3)*c6 - 0.088*s2*s3*s4*s6 + 0.21*s2*s3*s4*c6 - 0.384*s2*s3*s4 - 0.0825*s2*s3*c4 + 0.0825*s2*s3, 0.21*(s2*s4*c3 + c2*c4)*s6*c5 + 0.088*(s2*s4*c3 + c2*c4)*c5*c6 + 0.088*(s2*c3*c4 - s4*c2)*s6 - 0.21*(s2*c3*c4 - s4*c2)*c6 - 0.0825*s2*s4*c3 + 0.384*s2*c3*c4 - 0.384*s4*c2 - 0.0825*c2*c4, 0.21*(-(-s2*c3*c4 + s4*c2)*s5 + s2*s3*c5)*s6 + 0.088*(-(-s2*c3*c4 + s4*c2)*s5 + s2*s3*c5)*c6, -0.088*((-s2*c3*c4 + s4*c2)*c5 + s2*s3*s5)*s6 + 0.21*((-s2*c3*c4 + s4*c2)*c5 + s2*s3*s5)*c6 + 0.21*(s2*s4*c3 + c2*c4)*s6 + 0.088*(s2*s4*c3 + c2*c4)*c6, 0], [0, -s1, s2*c1, s1*c3 + s3*c1*c2, -(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4, ((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*s5 - (-s1*c3 - s3*c1*c2)*c5, (((-s1*s3 + c1*c2*c3)*c4 + s2*s4*c1)*c5 + (-s1*c3 - s3*c1*c2)*s5)*s6 - (-(-s1*s3 + c1*c2*c3)*s4 + s2*c1*c4)*c6], [0, c1, s1*s2, s1*s3*c2 - c1*c3, -(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4, ((s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*s5 - (-s1*s3*c2 + c1*c3)*c5, (((s1*c2*c3 + s3*c1)*c4 + s1*s2*s4)*c5 + (-s1*s3*c2 + c1*c3)*s5)*s6 - (-(s1*c2*c3 + s3*c1)*s4 + s1*s2*c4)*c6], [1, 0, c2, -s2*s3, s2*s4*c3 + c2*c4, (-s2*c3*c4 + s4*c2)*s5 - s2*s3*c5, ((-s2*c3*c4 + s4*c2)*c5 + s2*s3*s5)*s6 - (s2*s4*c3 + c2*c4)*c6]])


    return J



def calcJacobian2(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    fk = FK()
    jointPositions,T0e =    fk.forward(q_in)

    translation_vector = T0e[0:3,-1]

    approaches = fk.get_approach_for_all()

    #Build the Jacobian
    for i in range(len(q_in)):
        J[0:3,i] =get_skew_symmetric_matrix(approaches[:,i])@(np.add(translation_vector,-jointPositions[i,:]))
        J[3:6,i] = approaches[:,i]

    return J

def calcGenJacobian(q_in,joint_no=7):
    '''
    Only calculate velocity Jacob
    joint_no: Assume that joint_no for base frame is 0, so it belongs to (1,7)
    q_in: You have to give all q_in because I won't change FK for this
    '''

    J = np.zeros((3,joint_no))
    fk = FK()
    jointPositions,T0e = fk.forward(q_in)

    #This is our T0e
    translation_vector = jointPositions[joint_no-1,:]

    # translation_vector = T0e[0:3,-1]

    #These are approaches
    approaches = fk.get_approach_for_all()


    for i in range(joint_no-1):
        J[0:3,i] =get_skew_symmetric_matrix(approaches[:,i])@(np.add(translation_vector,-jointPositions[i,:]))

    return J


def get_skew_symmetric_matrix(vector):
    a = vector[0]
    b = vector[1]
    c = vector[2]

    return np.array([[0,-c,b],
                    [c,0,-a],
                    [-b,a,0]])




	



if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    # q = np.array([0,0,0,0,0,0,0])
    print(np.round(calcJacobian(q),3))
