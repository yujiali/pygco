import numpy as np
import pygco
import matplotlib.pyplot as plt

def test_general():
    gc = pygco.gco()
    gc.createGeneralGraph(n_sites, n_labels, True)
    print gc.handle
    gc.destroyGraph()

    u = np.array([
        [2, 8, 8], 
        [7, 3, 7], 
        [8, 8, 2],
        [6, 4, 6]
    ], dtype=np.intc)
    e = np.array([
        [0, 1],
        [1, 2],
        [2, 3]
    ], dtype=np.intc)
    ew = np.array([3, 10, 1], dtype=np.intc)
    s = (1 - np.eye(3)).astype(np.intc)

    print pygco.cut_general_graph(e, ew, u, s, n_iter=1)

def test_float():
    unary_cost = np.array([[0.0, 1.0, 2.0],
                           [4.0, 1.0, 0.0],
                           [0.0, 1.0, 2.0]])
    edges = np.array([[0, 1],
                      [1, 2],
                      [0, 2]]).astype(np.int32)
    pairwise_cost = np.array([[0.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0]])
    edge_weights = np.array([2.0, 0.0, 0.0])

    n_sites = 3
    n_labels = 3
    n_edges = 3

    print pygco.cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost, n_iter=-1, algorithm="swap")

def test_grid():
    x = np.zeros((100, 100))
    x[:,50:] = 2
    x[25:75, 25:75] = 1

    y = x + np.random.randn(100, 100)

    unary = np.tile(y[:,:,np.newaxis], [1,1,3])

    im = unary[:,:,1]
    im = im - 1
    im[x == 0] *= -1
    unary[:,:,1] = im
    unary[:,:,2] = 2 - unary[:,:,2]

    plt.ion()
    plt.figure(); plt.imshow(unary[:,:,0], cmap="gray", interpolation="nearest")
    plt.figure(); plt.imshow(unary[:,:,1], cmap="gray", interpolation="nearest")
    plt.figure(); plt.imshow(unary[:,:,2], cmap="gray", interpolation="nearest")

    pairwise = 1 - np.eye(3)
    z = pygco.cut_grid_graph_simple(unary, pairwise * 2, n_iter=-1)
    plt.figure(); plt.imshow(z.reshape(100, 100), cmap="gray", interpolation="nearest")
    
    t = raw_input('[Press any key]')

def test_binary():
    img = np.random.randn(100, 100)
    img[25:75,25:75] += 2
    img -= 1

    # !!! Be careful when doing this concatenation, it seems 'c_' does not create a copy
    #u = np.c_[img.flatten().copy(), -img.flatten().copy()]
    u = np.c_[img.reshape(10000,1), -img.reshape(10000,1)].copy()
    plt.ion()
    plt.figure(); plt.imshow(u[:,0].reshape(100,100), cmap='gray', interpolation='nearest')
    plt.figure(); plt.imshow(u[:,1].reshape(100,100), cmap='gray', interpolation='nearest')

    import imgtools.pairwise as pw
    edges, edge_weights = pw.get_uniform_smoothness_pw_single_image(img.shape)
    pw_cost = 1 - np.eye(2)

    # y = pygco.cut_general_graph(edges, edge_weights, u, pw_cost*2, n_iter=-1, algorithm='expansion')
    unary = np.tile(img[:,:,np.newaxis], [1,1,2])
    unary[:,:,0] = img
    unary[:,:,1] = -img

    unary_new = u.reshape((100,100,2))

    print np.abs(unary - unary_new).max()
    print (unary != unary_new).any()

    #import ipdb
    #ipdb.set_trace()

    #plt.figure(); plt.imshow(unary[:,:,0], cmap='gray', interpolation='nearest')
    #plt.figure(); plt.imshow(unary[:,:,1], cmap='gray', interpolation='nearest')
    #y = pygco.cut_grid_graph_simple(unary, pw_cost*0, n_iter=-1)
    #y_new = pygco.cut_grid_graph_simple(unary_new + np.random.randn(unary.shape[0], unary.shape[1], unary.shape[2])*0, pw_cost*0, n_iter=-1)
    y_new = pygco.cut_grid_graph_simple(unary_new + np.zeros(unary_new.shape), pw_cost*1, n_iter=-1)
    y_new2 = pygco.cut_grid_graph_simple(unary_new, pw_cost*0, n_iter=-1)

    print unary_new.dtype
    print (unary_new + np.zeros(unary_new.shape)).dtype
    
    #plt.figure(); plt.imshow(y.reshape(100,100), cmap='gray', interpolation='nearest')
    plt.figure(); plt.imshow(y_new.reshape(100,100), cmap='gray', interpolation='nearest')
    plt.figure(); plt.imshow(y_new2.reshape(100,100), cmap='gray', interpolation='nearest')

    t = raw_input('[Press any key]')

if __name__ == "__main__":
    # test_grid()
    test_binary()

