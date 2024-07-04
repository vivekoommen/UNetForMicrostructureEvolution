import sys
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
tf.config.experimental.enable_tensor_float_32_execution(True)

from unet import unet

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
        pass

@tf.function()
def train_step(model, x,dt, y, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x,dt)
        loss   = model.loss(y_pred, y)[0]

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return(loss)

def preprocess(traj, Par):
    # traj - [_, 14, 128, 128, 3]
    x = sliding_window_view(traj[:,:-Par['lf'],:,:,0:1], window_shape=Par['lb'], axis=1 ).transpose(0,1,2,3,5,4).reshape(-1,Par['nx'], Par['ny'], Par['lb'], 1)
    y = sliding_window_view(traj[:,Par['lb']:,:,:,0:1], window_shape=Par['lf'], axis=1 ).transpose(0,1,5,2,3,4).reshape(-1,Par['lf'],Par['nx'], Par['ny'], 1)
    print('x: ', x.shape)
    print('y: ', y.shape)
    return x,y

def main():
    np.random.seed(23)
    tensor = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

    train = np.load('../data/sample_train_data.npy')
    val   = np.load('../data/sample_val_data.npy') 
    test  = np.load('../data/sample_test_data.npy')
    MIN   = np.load('../data/MIN.npy')
    MAX   = np.load('../data/MAX.npy')

    Par = {}
    Par['nt'] = train.shape[1]
    Par['nx'] = train.shape[2]
    Par['ny'] = train.shape[3]
    Par['nf'] = train.shape[4]-1

    Par['lb'] = 3
    Par['lf'] = 9
    Par['temp'] = Par['nt'] - Par['lb'] - Par['lf'] + 1

    print('\nTrain Dataset')
    x_train, y_train = preprocess(train, Par)
    print('\nVal Dataset')
    x_val, y_val   = preprocess(val,  Par)
    print('\nTest Dataset')
    x_test, y_test   = preprocess(test,  Par)

    Par['inp_shift'] = MIN
    Par['inp_scale'] = MAX-MIN
    Par['out_shift'] = MIN
    Par['out_scale'] = MAX-MIN

    num_samples = x_train.shape[0]
    dt = np.linspace(0,1,Par['lf']).reshape(-1,1)
    print('dt: ', dt.shape)

    address = 'unet'
    Par['address'] = address

    print('shuffling train dataset')
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    print('shuffling complete')

    model = unet(Par)
    _ = model( tensor(x_train[0:1]), tensor(dt))
    print(model.summary())
 
    model_number = 0

    n_epochs = 10
    train_batch_size = 4
    val_batch_size   = 1
    test_batch_size  = 1
    optimizer = tf.keras.optimizers.Adam(learning_rate = 2*10**-4)

    lowest_loss = 1000
    begin_time = time.time()
    print('Training Begins')
    first=True
    
    for i in range(model_number+1, n_epochs+1):
        for j in np.arange(0, num_samples-train_batch_size+1, train_batch_size):
            loss = train_step(model, tensor(x_train[j:(j+train_batch_size)]), tensor(dt), tensor(y_train[j:(j+train_batch_size)]), optimizer)
        if i%1 == 0:

            train_loss = loss.numpy()

            val_loss_ls=[]
            for k in range(0, x_val.shape[0]-val_batch_size+1, val_batch_size):
                y_pred = model(x_val[k:k+val_batch_size],dt)
                loss   = model.loss(y_pred, y_val[k:k+val_batch_size])[0].numpy()
                val_loss_ls.append(loss)

            val_loss = np.mean(val_loss_ls)

            if val_loss<lowest_loss:
                lowest_loss = val_loss
                model.save_weights(address + "/model")

            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            model.index_list.append(i)
            model.train_loss_list.append(train_loss)
            model.val_loss_list.append(val_loss)

    print('Training complete')

    test_loss_ls=[]
    for k in range(0, x_test.shape[0]-test_batch_size+1, test_batch_size):
        y_pred = model(x_test[k:k+test_batch_size],dt)
        loss   = model.loss(y_pred, y_test[k:k+test_batch_size])[0].numpy()
        test_loss_ls.append(loss)

    test_loss = np.mean(test_loss_ls)
    print("Test Loss: " + "{:.3e}".format(test_loss) )

    #Convergence plot
    index_list = model.index_list
    train_loss_list = model.train_loss_list
    val_loss_list = model.val_loss_list
    np.savez(address+'/convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)

    plt.close()
    fig = plt.figure(figsize=(10,7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="val", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig( address + "/convergence.png", dpi=200)
    plt.close()
    print('--------Complete--------')

main()
