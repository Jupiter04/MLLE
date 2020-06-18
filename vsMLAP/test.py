import numpy as np

# clu1 = np.random.randn(10, 3)*0.1 + 2
# clu2 = np.random.randn(10, 3)*0.1 - 2
# clu3 = np.random.randn(10, 3)*0.1
#
# data = np.concatenate((clu1,clu2,clu3), 0)
#
# k = 3
#
# n = data.shape[0]
# p = data.shape[1]
# centers = np.zeros((k,p))
# for i in range(p):
#     feature_min = np.min(data[:,i])
#     feature_max = np.max(data[:,i])
#     centers[:,i] = feature_min + np.random.rand(k)*(feature_max-feature_min)
#
# index = np.zeros((n,1))
# f = True
# while(f):
#     f = False
#     for j in range(n):
#         dis_min = np.inf
#         index_min = index[j]
#         for i in range(k):
#             dis = np.sum(np.power(data[j,:]-centers[i,:] , 2))
#             if dis<dis_min:
#                 dis_min = dis
#                 index_min = i
#         if(index_min!=index[j]):
#             f = True
#             index[j] = index_min
#     for i in range(k):
#         data_in = data[np.where(index==i)[0], :]
#         if(data_in.shape[0]!=0):
#             centers[i,:] = np.mean(data_in, 0)
#
# print(index)
# print(centers)




clu1 = np.random.randn(10,2)*0.1 + np.array([1,1])
clu2 = np.random.randn(10,2)*0.1 + np.array([-1,-1])
clu3 = np.random.randn(10,2)*0.1 + np.array([0,0])

data = np.concatenate((clu1,clu2,clu3), 0)

n = data.shape[0]
p = data.shape[1]

k = 3

centers = np.zeros((k,p))
index = np.zeros((n,1))
for i in range(p):
    min_feature = np.min(data[:,i])
    max_feature = np.max(data[:,i])
    centers[:,i] = min_feature + np.random.rand(k)*(max_feature-min_feature)

f = True

while(f):
    f = False
    for i in range(n):
        dis_min = np.inf
        ind = 0
        for j in range(k):
            dis = np.sqrt(np.sum(np.power(data[i,:]-centers[j,:], 2)))
            if(dis_min > dis):
                dis_min = dis
                ind = j
        if(ind!=index[i]):
            f = True
            index[i] = ind

    for j in range(k):
        data_clu = data[np.squeeze(index==j),:]
        if(data_clu.shape[0]!=0):
            centers[j,:] = np.mean(data_clu, 0)

print(index)
print(centers)






