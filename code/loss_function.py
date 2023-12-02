from math import sin,pi,cos,exp
import torch
import numpy as np
from data_loader import batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def N(n,k):
    N_1=torch.tensor([n+k*1j])
    N_1=N_1.to(device)
    # print(N_1)
    return N_1
def rp_1_2(N1,N2):#x1
    N1=torch.tensor(N1)
    # N2 = torch.tensor([N2.real, N2.imag])
    theta1=torch.tensor((64.5/180)*pi)
    theta2=torch.arcsin(torch.sin(theta1)/N2)
    # print(theta2)
    rp_1_2=(N2*torch.cos(theta1)-torch.cos(theta2))/(N2*torch.cos(theta1)+torch.cos(theta2))
    # print((N2*torch.cos(theta1)+torch.cos(theta2)))
    return rp_1_2
def rp_2_3(N2,N3):#x2
    # N2 = torch.tensor([N2.real, N2.imag])
    # N3 = torch.tensor([N3.real, N3.imag])
    theta1=torch.tensor((64.5/180)*pi)
    theta2=torch.arcsin(torch.sin(theta1)/N2)
    theta3=torch.arcsin(torch.sin(theta1)/N3)
    rp_2_3=(N3 * torch.cos(theta2) - N2 * torch.cos(theta3)) / (N3 * torch.cos(theta2) + N2 * torch.cos(theta3))
    return rp_2_3
def rs_1_2(N1,N2):#y1
    theta1=torch.tensor((64.5/180)*pi)
    theta2=torch.arcsin(torch.sin(theta1)/N2)
    rs_1_2=(N1*torch.cos(theta1)-N2*torch.cos(theta2))/(N1*torch.cos(theta1)+N2*torch.cos(theta2))
    return rs_1_2
def rs_2_3(N2,N3):#y2
    theta1=torch.tensor((64.5/180)*pi)
    theta2=torch.arcsin(torch.sin(theta1)/N2)
    theta3=torch.arcsin(torch.sin(theta1)/N3)
    rs_2_3=(N2*torch.cos(theta2)-N3*torch.cos(theta3))/(N2*torch.cos(theta2)+N3*torch.cos(theta3))
    return rs_2_3
def t(N2,lamda,d):
    lamda=torch.tensor(lamda)
    d=torch.tensor(d)
    lamda=lamda.to(device)
    d=d.to(device)
    theta1=torch.tensor((64.5/180)*pi)
    theta1=theta1.to(device)
    theta2=torch.arcsin(torch.sin(theta1)/N2)
    theta2=theta2.to(device)
    beta = 2 * pi * (d / lamda) * N2 * torch.cos(theta2)
    beta=beta.to(device)
    t=torch.exp(torch.tensor(0+(-2)*beta*1j))
    return t
def result(a,b):
    result=torch.tan(a)*torch.exp(torch.tensor(0+b*1j))
    return result
def loss_1(y_predict,y_true,x_true,csv_data):
    # csv_data=np.array(csv_data)
    y_predict=torch.tensor(y_predict,requires_grad=True)
    loss=0
    for i in range(batch_size):
        n2=y_predict[i][0]
        k2=y_predict[i][1]
        n3=y_predict[i][2]
        k3=y_predict[i][3]
        d=y_predict[i][4]
        x1=x_true[i][0]
        x2=x_true[i][1]
        n2_1=y_true[i][0]
        k2_1=y_true[i][1]
        n3_1=y_true[i][2]
        k3_1=y_true[i][3]
        d_1=y_true[i][4]
        mask = (csv_data[:, 3] == n2_1) & (csv_data[:, 4] == k2_1) & (csv_data[:, 5] == n3_1) & (csv_data[:, 6] == k3_1) & (csv_data[:, 7] == d_1)
        lamda = csv_data[mask, 0][0]
        N2 = N(n2, k2)
        N3 = N(n3, k3)
        rp12 = rp_1_2(1, N2)
        rp23 = rp_2_3(N2, N3)
        rs12 = rs_1_2(1, N2)
        rs23 = rs_2_3(N2, N3)

        t_value = t(N2, lamda, d)
        rpp=(rp12 + rp23 * t_value) / (1 + rp12 * rp23 * t_value)
        rss=(rs12 + rs23 * t_value) / (1 + rs12 * rs23 * t_value)
        result_cal = rpp/rss
        result_true = result(x1, x2)
        loss += torch.square(result_cal.real - result_true.real) + torch.square(result_cal.imag - result_true.imag)
    loss=loss/batch_size
    return loss

def acc(predicted_vectors, true_vectors):
    predicted_vectors = np.array(predicted_vectors)
    true_vectors = np.array(true_vectors)
    cosine_similarities = np.sum(predicted_vectors * true_vectors, axis=1) / (np.linalg.norm(predicted_vectors, axis=1) * np.linalg.norm(true_vectors, axis=1))
    # 计算准确率的平均值
    accuracy = np.mean(cosine_similarities)
    return accuracy
