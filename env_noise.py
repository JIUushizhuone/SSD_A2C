import numpy as np
import math
Data = np.load('/home/jiushizhuone/TFagents/env_data/sy5.0G.npy')


def Rayleigh_Noise_single(dimension, Rayleigh_factor = 1.732):
    return abs((Rayleigh_factor*np.random.randn(dimension)
                +Rayleigh_factor*1j*np.random.randn(dimension))/ np.sqrt(2))

def Rayleigh_Noise(data, Rayleigh_factor = 1.732):
    a = np.array(data); 
    array_size = np.array(data.shape)
    if  a.size == 3:
        Rayleigh_ch =abs((Rayleigh_factor*np.random.randn(array_size[0],array_size[1],array_size[2])
                +Rayleigh_factor*1j*np.random.randn(array_size[0],array_size[1],array_size[2]))/ np.sqrt(2))
    else:
        Rayleigh_ch =abs((Rayleigh_factor*np.random.randn(array_size[0],array_size[1],array_size[2],array_size[3])
                    +Rayleigh_factor*1j*np.random.randn(array_size[0],array_size[1],array_size[2],array_size[3]))/ np.sqrt(2))
    return Rayleigh_ch + data

def System_bias(data = Data, bias = 3):
    new_data = data + bias
    return new_data

# new_data = Rayleigh_Noise(data)

# a: distance between pre_loc and next_loc
# b: distance between pre_loc and Tx
# c: distance between next_loc and Tx
def pos_2_angle(a,b,c):
    # print(a,b,c)
    if a+b >c and a+c >b and b+c >a:
        pi = math.pi
        A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c))); ratio_A = abs(math.cos(A*pi/180))
        B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c))); ratio_B = abs(math.cos(B*pi/180))
        C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b))); ratio_C = abs(math.cos(C*pi/180))
    else:
        ratio_A = 0; ratio_B = 1; ratio_C = 1
    return ratio_A, ratio_B, ratio_C



def Device_shadowing(scenario_id, Tx_id, prev_loc, next_loc, prev_RSSI, next_RSSI, shadow_loss = 2.0):

    Tx_loc_Siyuan = np.array([[27,9,3],[38,8,3],[26,14,7],[39,14,7],[29,4,7],[39,4,7],[27,4,10],[40,4,10]]) - [11,1,0] - 1
    Tx_loc_Xuehuo = np.array([[19,1,6],[8,4,5],[24,8,7],[3,12,8],[1,6,7],[11,8,8],[3,-3,7],[2,-1,2],[7,-16,3],[32,7,6]]) - 1
    
    if scenario_id == 0: 
        Tx_loc = Tx_loc_Siyuan # 0 represents Siyuan
    elif scenario_id == 1:
        Tx_loc = Tx_loc_Xuehuo # 1 represents Xuehuo
    else:
        print('Error: Wrong data shape!')
        return 0

    for i in np.arange(Tx_id.shape[0]):
        if sum(abs(next_loc-Tx_loc[Tx_id[i],0:2])) - sum(abs(prev_loc-Tx_loc[Tx_id[i],0:2])) > 0: # The UE is leaving far away from Tx
            a = np.sqrt((next_loc[0]-prev_loc[0])**2 + (next_loc[1]-prev_loc[1])**2)
            b = np.sqrt((prev_loc[0]-Tx_loc[Tx_id[i],0])**2 + (prev_loc[1]-Tx_loc[Tx_id[i],1])**2)
            c = np.sqrt((next_loc[0]-Tx_loc[Tx_id[i],0])**2 + (next_loc[1]-Tx_loc[Tx_id[i],1])**2)
            ratio_A, ratio_B, ratio_C = pos_2_angle(a,b,c)
            next_RSSI[i] +=  -(ratio_B * shadow_loss)
            prev_RSSI[i] +=  -(ratio_C * shadow_loss)
        else:
            pass
    
    return prev_RSSI, next_RSSI

def data_preprocessing(data, RSSI_id =[0,1,2,4,6]):
    RSSI_id = np.array(RSSI_id)
    for i in np.arange(data.shape[0]):
        mu = np.mean(data[i,:,:])
        sigma = np.std(data[i,:,:])
        data[i,:,:] = (data[i,:,:] - mu)/sigma
        data[i,:,:] = data[i,:,:] - np.min(data[i,:,:])
    return data[RSSI_id,:,:]


if __name__=='__main__':
    # new_data =  System_bias(Rayleigh_Noise(Data))
    test_data1 = np.array([10,10,10,10,10,10,10,10],float)
    test_data2 = np.array([10,10,10,10,10,10,10,10],float)
    # test_data1, test_data2 = Device_shadowing(new_data, 0, [[15,9,3]], [14,16,4], test_data1, test_data2)