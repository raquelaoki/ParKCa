from os import listdir
from os.path import isfile, join

def deconfounder(train,colnames,y01,type,k):
    #df = pd.DataFrame(np.arange(1,10).reshape(3,3))
    #arr = sparse.coo_matrix(([1,1,1], ([0,1,2], [1,2,0])), shape=(3,3))
    #df['newcol'] = arr.toarray().tolist()
    W,F = fm_MF(train,k)
    pred  = check_save(W,train,colnames,y01,'MF',type,k)

    pca = fm_PCA(train,k)
    pred = check_save(pca,train,colnames,y01,'PCA',type,k)

    ac = fm_A(train,k)
    pred = check_save(pca,train,colnames,y01,'A',type,k)


        #only full
    return outputs

def fm_MF(train,k):
    '''
    Matrix Factorization to extract latent features
    Parameters:
        train: dataset
        k: latent Dimension
    Return:
        2 matrices
    '''
    model = NMF(n_components=k, init='random') #random_state=0
    W = model.fit_transform(train)
    H = model.components_

    return W, H

def fm_PCA(train,k):
    '''
    PCA to extrac latent features
    Parameters:
        train: dataset
        k: latent Dimension
    Return:
        1 matrix
    '''
    X = StandardScaler().fit_transform(train)
    model = PCA(n_components=k)
    principalComponents = model.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)

    return principalDf

def fm_A(train,k):
    from keras.layers import Input, Dense
    from keras.models import Model
    '''
    Autoencoder to extrac latent features
    Parameters:
        train: dataset
        k: latent Dimension
        run: True/False
    Return:
        1 matrix
    References
    #https://www.guru99.com/autoencoder-deep-learning.html
    #https://blog.keras.io/building-autoencoders-in-keras.html
    '''
    x_train, x_test = train_test_split(train, test_size = 0.3,random_state = 22)
    print(x_train.shape, x_test.shape, train.shape)
    ii = x_train.shape[1]
    input_img = Input(shape=(ii,))
    encoding_dim = 20
    encoded = Dense(encoding_dim, activation='sigmoid')(input_img) #change relu
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(ii, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)


    autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(train)
    return encoded_imgs

def predictive_check(X,Z):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    '''
    This function is agnostic to the method.
    Use a Linear Model X_m = f(Z), save the proportion
    of times that the pred(z_test)<X_m(test) for each feature m.
    Compare with the proportion of the null model mean(x_m(train)))<X_m(test)
    Create an Confidence interval for the null model, check if the average value
    across the predicted values using LM is inside this interval

    Sample a few columns (300 hundred?) to do this math

    Parameters:
        X: orginal features
        Z: latent (either the reconstruction of X or lower dimension)
    Return:
        v_obs values and result of the test
    '''
    #If the number of columns is too large, select a subset of columns instead
    if X.shape[1]>10000:
        X = X[:,np.random.randint(0,X.shape[1],10000)]

    v_obs = []
    v_nul = []
    for i in range(X.shape[1]):
        Z_train, Z_test, X_train, X_test = train_test_split(Z, X[:,i], test_size=0.3)
        model = LinearRegression().fit(Z_train, X_train)
        X_pred = model.predict(Z_test)
        v_obs.append(np.less(X_test, X_pred).sum()/len(X_test))
        v_nul.append(np.less(X_test, X_train.mean(),).sum()/len(X_test))

    #Create the Confidence interval
    n = len(v_nul)
    m, se = np.mean(v_nul), np.std(v_nul)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    if m-h<= np.mean(v_obs) and np.mean(v_obs) <= m+h:
        return v_obs, True
    else:
        return v_obs, False
