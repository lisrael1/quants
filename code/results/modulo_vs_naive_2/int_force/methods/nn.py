import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
import int_force

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    samples, number_of_bins, quant_size=100, 1024, 0.001
    snr=1000001
    mod_size = number_of_bins * quant_size

    df=pd.DataFrame()
    print('generating data')
    for i in tqdm(range(10000)):
        cov = int_force.rand_data.rand_data.rand_cov(snr=snr)
        data = int_force.rand_data.rand_data.random_data(cov, samples)
        '''modulo and quantization'''
        tmp = int_force.methods.methods.sign_mod(data, mod_size)
        tmp = int_force.methods.methods.to_codebook(tmp, quant_size, 0)
        tmp = int_force.methods.methods.from_codebook(tmp, quant_size, 0)
        tmp.columns = [['after'] * 2, tmp.columns.values]
        data.columns = [['before'] * 2, data.columns.values]
        data = data.join(tmp)
        del tmp
        tmp=data.stack().stack().reset_index(1, drop=True).to_frame().swaplevel(1, 0, axis=0).sort_index().T
        df=df.append(tmp, ignore_index=True)

    learn, test = np.split(df, [int(df.shape[0] * 0.8)])
    test.reset_index(drop=True, inplace=True)

    if 0:
        import pylab as plt
        fig = plt.figure()
        fig.suptitle('data')
        data.after.set_index('X').Y.plot(style='.', alpha=0.1, label='after')
        data.before.set_index('X').Y.plot(style='.', alpha=0.1, label='original')
        fig.legend()
        plt.show()

    print('generating net')
    from keras.models import Sequential
    import keras
    from keras.layers import Dense, Activation
    from glob import glob
    from keras_tqdm import TQDMNotebookCallback, TQDMCallback

    ################# defining the NN without data #################
    if len(glob('model_checkpoint.h5')):
        print('trained model is saved, loading it')
        from keras.models import load_model

        model = load_model('model_checkpoint.h5')
    else:
        model = Sequential()
        model.add(Dense(samples*10, input_dim=samples*2))
        model.add(Activation('relu'))
        model.add(Dense(samples*20))
        model.add(Activation('relu'))
        model.add(Dense(samples*10))
        model.add(Activation('relu'))
        model.add(Dense(units=samples*2))

        model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['mse'])

    print('training net')
    ################# learn - just enter input output #################
    epochs=100
    epochs=100
    epochs=2000

    batch_size=256
    batch_size=1024

    checkpoint=keras.callbacks.ModelCheckpoint('model_checkpoint.h5', monitor='val_loss',
                                               verbose=0, save_best_only=False, save_weights_only=False,
                                               mode='auto', period=1)
    hist=keras.callbacks.CSVLogger('history.csv', separator=',', append=True)

    reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_mean_squared_error', factor=0.6,
                                                patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    history = model.fit(learn.after.values, learn.before.values,
                        validation_data=(test.after.values, test.before.values), batch_size=batch_size, epochs=epochs, verbose=1,
                      callbacks=[reduce_lr, checkpoint, hist])

    import plotly as py
    import cufflinks

    history = pd.read_csv('history.csv', index_col=None)
    fig = history.figure(xTitle='epoch', title='learning and validation convergance')
    py.offline.iplot(fig)
    history