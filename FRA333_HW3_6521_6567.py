# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส(ธนวัฒน์_6461)
1.ณัชณศา_6521
2.ชัญญาภัค_6567
3.
'''
# ==============================================================================================================#
#Import Library
import numpy as np
import HW3_utils

#=============================================<คำตอบข้อ 1>======================================================#
def endEffectorJacobianHW3(q:list[float])->list[float]:

    #Get Rotation Matrix and Traslation Matrix from HW3_utils.py
    R,P,R_e,p_e = HW3_utils.FKHW3(q)

    #Create Jacobian Matrix (6 row and 3 column)
    J_e = np.empty((6,3))

    for i in range(len(q)):

        #Jacobian linear velocity 
        J_v = np.cross(R[:,:,i][:,2],(p_e - P[:,i])) 

        #Jacobian angular velocity
        J_w = R[:,:,i][:,2]     

        #Join matrix of linear velocity and angular velocity                          
        J_i = np.concatenate((J_v,J_w),axis=0)       

        #Append Jacobian of each joint
        J_e[:,i] = J_i   

    return J_e
#==============================================================================================================#

#=============================================<คำตอบข้อ 2>======================================================#
def checkSingularityHW3(q:list[float])->bool:

    J_q = endEffectorJacobianHW3(q)

    #Part of linear velocity(3x3 matrix)
    J_v_part = J_q[:3, :]

    #Determinant of Jacobian
    determinant = np.linalg.det(J_v_part)

    if abs(determinant) < 1e-3:
        #Near Singularity 
        return True
    else:
        #Non Singularity
        return False
#==============================================================================================================#

#=============================================<คำตอบข้อ 3>======================================================#
def computeEffortHW3(q: list[float], w: list[float]) -> list[float]:

    J = endEffectorJacobianHW3(q)
    
    R, P, R_e, p_e = HW3_utils.FKHW3(q)

    # Force
    f_e = w[3:]
    f_0 = R_e @ f_e  # Transform force to base frame

    # Moment
    n_e = w[:3]  
    n_0 = R_e @ n_e  # Transform moment to base frame

    # Concatenate force and moment to create wrench
    w_0 = np.concatenate((f_0, n_0), axis=0)
    
    # Transpose of the Jacobian
    J_t = np.transpose(J)
    
    # Calculate joint torque
    tau = J_t @ w_0
    return tau
#==============================================================================================================#