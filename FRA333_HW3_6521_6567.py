# file สำหรับเขียนคำตอบ
# ในกรณีที่มีการสร้าง function อื่น ๆ ให้ระบุว่า input-output คืออะไรด้วย
'''
ชื่อ_รหัส(ธนวัฒน์_6461)
1.ณัชณศา_6521
2.ชัญญาภัค_6567
3.
'''
#Import Library
import numpy as np
import HW3_utils

#=============================================<คำตอบข้อ 1>======================================================#
#code here
def endEffectorJacobianHW3(q:list[float])->list[float]:
    R,P,R_e,p_e = HW3_utils.FKHW3(q)
    J_q = np.empty((6,3))
    for i in range(len(q)):

        #Jacobian linear velocity 
        J_v = np.cross(R[:,:,i][:,3],(p_e - P[:,i])) 

        #Jacobian angular velocity
        J_w = R[:,:,i][:,2]     

        #Join matrix of linear velocity and angular velocity                          
        J_i = np.concatenate((J_v,J_w),axis=0)       

        #Append Jacobian of each joint
        J_q[:,i] = J_i   

    return J_q
print(endEffectorJacobianHW3([0,0,0]))


#
#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:
    pass
#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:
    pass
#==============================================================================================================#