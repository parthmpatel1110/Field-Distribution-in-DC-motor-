import numpy as np 
import csv
import matplotlib.pyplot as plt



def load_data(file):
    w = []
    with open(file,'r') as f:
            reader = csv.reader(f,delimiter='\n')
            for row in reader:
                w = w + row
    return w

def load_no_plane(data):
    node_start_index = data.index("$PhysicalNames") + 2
    node_count = int(data[node_start_index - 1])
   
    return node_count

def load_nodes(data):
    node_start_index = data.index("$Nodes") + 2
    node_count = int(data[node_start_index - 1])
    node_end_index = node_start_index + node_count
    
    T = data[node_start_index:node_end_index]

    T = np.array([[float(j) for j in i.split(' ')] for i in T])
    
    return T[:,1],T[:,2]


def load_edges_and_triangles(data):
    element_start_index = data.index("$Elements") + 2
    element_count = int(data[element_start_index - 1])
    element_end_index = element_start_index + element_count

    Ele = data[element_start_index:element_end_index]
    Ele = [[int(j) for j in i.split(' ')] for i in Ele]

    edge_count = 0
    for a in Ele:
        if len(a) > 7:
            break
        edge_count += 1
    
    E = Ele[:edge_count]
    T = Ele[edge_count:]
    
    E = np.array(E)
    T = np.array(T)

    return E[:,[3,5,6]], T[:,[3,5,6,7]]


def calc_C(X,Y,E,T):
    no_ele = T.shape[0]
    no_node = X.shape[0]
    print("number of element=",no_ele)

    x,y,e,t = X,Y,E,T

    P = np.zeros((no_ele,3))
    Q = np.zeros((no_ele,3))

    P[:,0] = y[t[:,2]-1] - y[t[:,3]-1]
    P[:,1] = y[t[:,3]-1] - y[t[:,1]-1]
    P[:,2] = y[t[:,1]-1] - y[t[:,2]-1]

    Q[:,0] = x[t[:,3]-1] - x[t[:,2]-1]
    Q[:,1] = x[t[:,1]-1] - x[t[:,3]-1]
    Q[:,2] = x[t[:,2]-1] - x[t[:,1]-1]

    delta = 0.5*abs(P[:,1]*Q[:,2] - P[:,2]*Q[:,1])
    mu = np.ones(no_ele)

    for i in range(no_ele):
        if t[i,0]==1 or t[i,0]==2:
            mu[i]=1000 

    c = np.array([(P*P[:,i].reshape(P.shape[0],1) + Q*Q[:,i].reshape(Q.shape[0],1)) for i in range(P.shape[1])]) 
    c = np.array([(c[:,i,:])/(4*delta[i]*mu[i]) for i in range(delta.shape[0])])

    C =np.zeros((no_node,no_node))
    for a in range(no_ele):
        nodes= t[a,1:4]
        for i in  range(3):
            for j in range(3):
                C[nodes[i]-1,nodes[j]-1] = C[nodes[i]-1,nodes[j]-1] + c[a,i,j]

            
    for i in range(e.shape[0]):
        if e[i,0] == 8:
            C[e[i,1]-1,:]=0
            C[e[i,2]-1,:]=0
            C[e[i,1]-1,e[i,1]-1]=1
            C[e[i,2]-1,e[i,2]-1]=1
    return C

def calculate_B(X,Y,E,T,no_plane,S):
    no_ele = T.shape[0]
    no_node = X.shape[0]

    B = np.zeros(no_node)
    for i in range(no_ele):
        if T[i,0]==1:
            B[T[i,1:4]-1]=S[0]
        if T[i,0]==2:
            B[T[i,1:4]-1]=S[1]
        if T[i,0]==3:
            B[T[i,1:4]-1]=S[2]
        if T[i,0]==4:
            B[T[i,1:4]-1]=S[3]
        if T[i,0]==5:
            B[T[i,1:4]-1]=S[4]
        if T[i,0]==6:
            B[T[i,1:4]-1]=S[5]
        if T[i,0]==7:
            B[T[i,1:4]-1]=S[6]
        if T[i,0]==8:
            B[T[i,1:4]-1]=S[7]

    return B

    

# @jit(nopython=False)
def calc_potential_at_other_points(T,X,Y,n):
    # can be improved with matrix multiplication
    
    Ti_x = []
    Ti_y = []
    # Xi = np.array([])
    # Yi = np.array([])
    
    for i in range(T.shape[0]):
        t = T[i]
        x1,y1 = X[t[1]-1], Y[t[1]-1]
        x2,y2 = X[t[2]-1], Y[t[2]-1]
        x3,y3 = X[t[3]-1], Y[t[3]-1]

        # mat = np.array([
        #     [1, x1, y1],
        #     [1, x2, y2],
        #     [1, x3, y3],
        # ])
        
        # v = np.array([[V[t[1]]],[V[t[2]]],[V[t[3]]]])
        # v = v.reshape(3,1)
        # delta = np.dot(np.linalg.inv(mat),v)
        Tx,Ty = get_integer_point_inside_tri(x1*n, y1*n, x2*n, y2*n, x3*n, y3*n)
    
        Ti_x.append(Tx/n)
        Ti_y.append(Ty/n)
        
    Ti = np.array([Ti_x,Ti_y])
    



    #     for i in range(Tx.shape[0]):
    #         p = np.array([1, Tx[i], Ty[i]])
    #         v = np.dot(p,delta)[0]
    #         np.append(Xi, Tx[i])
    #         np.append(Yi, Ty[i])
    #         np.append(Vi, v)

    return Ti        

def get_integer_point_inside_tri(x1, y1, x2, y2, x3, y3):
    # https://stackoverflow.com/questions/37181829/determining-all-discrete-points-inside-a-triangle
    xs = np.array((x1,x2,x3),dtype=float)
    ys = np.array((y1,y2,y3),dtype=float)

    x_range = np.arange(np.min(xs),np.max(xs)+1, 1, dtype = int)
    y_range = np.arange(np.min(ys),np.max(ys)+1, 1, dtype = int)

    X,Y = np.meshgrid( x_range,y_range )
    xc = np.mean(xs)
    yc = np.mean(ys)

    triangle = np.ones(X.shape,dtype=bool)
    for i in range(3):
        ii = (i+1)%3
        if xs[i] == xs[ii]:
            include = X *(xc-xs[i])/abs(xc-xs[i]) > xs[i] *(xc-xs[i])/abs(xc-xs[i])
        else:
            poly = np.poly1d([(ys[ii]-ys[i])/(xs[ii]-xs[i]),ys[i]-xs[i]*(ys[ii]-ys[i])/(xs[ii]-xs[i])])
            include = Y *(yc-poly(xc))/abs(yc-poly(xc)) > poly(X) *(yc-poly(xc))/abs(yc-poly(xc))
        triangle *= include
    
    return X[triangle], Y[triangle]

def main():
    file = "motor_v2_5254.msh"
    data = load_data(file)
    X,Y = load_nodes(data)
    no_plane = load_no_plane(data)
    E,T = load_edges_and_triangles(data)
    C = calc_C(X,Y,E,T)
    np.save("C",C)
    n = 500
    Ti = calc_potential_at_other_points(T, X, Y, n)
    np.save("Ti",Ti)
    # plt.contour(Xi,Yi,Vi.T,10)


if __name__ == "__main__":
    main()


