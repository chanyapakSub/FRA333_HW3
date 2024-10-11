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
        J_v = np.cross(R[:,:,i][:,2],(p_e - P[:,i])) 

        #Jacobian angular velocity
        J_w = R[:,:,i][:,2]     

        #Join matrix of linear velocity and angular velocity                          
        J_i = np.concatenate((J_v,J_w),axis=0)       

        #Append Jacobian of each joint
        J_q[:,i] = J_i   

    return J_q
# print(endEffectorJacobianHW3([0,0,0]))

#==============================================================================================================#
#=============================================<คำตอบข้อ 2>======================================================#
#code here
def checkSingularityHW3(q:list[float])->bool:
    J_q = endEffectorJacobianHW3(q)
    # ใช้เฉพาะส่วนของ Jacobian ที่เกี่ยวข้องกับการเคลื่อนที่เชิงเส้น (3x3 matrix)
    J_v_part = J_q[:3, :]
    # คำนวณ Determinant ของ Jacobian ส่วนที่เกี่ยวข้องกับการเคลื่อนที่เชิงเส้น
    determinant = np.linalg.det(J_v_part)
    # ตรวจสอบว่า Jacobian มีค่า determinant เป็นศูนย์หรือไม่
    if abs(determinant) < 1e-3:  # กำหนด threshold ใกล้ศูนย์
        # print("อยู่ในจุด Singularity")
        return True
    # print("ไม่อยู่ในจุด Singularity")
    return False
# print(checkSingularityHW3([4.71830409,2.78974574, 0.69907425]))
#==============================================================================================================#
#=============================================<คำตอบข้อ 3>======================================================#
#code here
def computeEffortHW3(q:list[float], w:list[float])->list[float]:
    pass
#==============================================================================================================#