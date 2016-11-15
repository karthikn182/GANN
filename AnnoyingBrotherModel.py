

#Things to-do
#1. make output of Generator a sequence (not probabilties!)
#2. Figure out concatenating in the right manner!
#3. make sure train() function is correct

class AnnoyingBrotherConfig(object):
    """Define HyperParameters"""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    sequence_length = 5
    hidden_size = 50
    memory_dim = 100
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 1
    input_vocab_size = 500
    output_vocab_size = 502 #adding 2 to each word
    d_output_vocab = 1 #aka number of classes (in this case P(real data))

class ABModel(object):

    def __init__(self,config):
        #initialize model variables
        #Combine the generator and discriminator model
        self.lr = config.learning_rate
        self.lr_decay = config.lr_decay
        self.init_scale = config.init_scale
        self.num_layers = config.num_layers

        self.seq_length = config.sequence_length
        self.hidden_size = config.hidden_size
        self.memory_dim = config.memory_dim
        self.max_epoch = config.max_epoch
        self.keep_prob = config.keep_prob

        self.batch_size = config.batch_size
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size
        self.d_output_vocab = 2

        #INITIALIZING
        tf.reset_default_graph()

        #This initializer is used to initialize all the weights of the network (not done until train() is called)
        initializer = tf.truncated_normal_initializer(stddev=self.init_scale)

        #These two placeholders are used for input into the generator and discriminator, respectively.
        q_in = tf.placeholder(shape=[None,self.seq_length,1],dtype=tf.float32) #Question vector (None means any shape aka for batching,seq_length,word_dimension)
        real_in = tf.placeholder(shape=[None,2*self.seq_length,1],dtype=tf.float32) #Real "Question/Answer" placeholder (none,2*seq_length,1)

        #Note need some answer HERE!

        Gz = generator(q_in) #Generates answer from question input
        #Only create the Discriminator Variables once!
        with tf.variable_scope("generator") as scope:
            Dx = discriminator(real_in) #Produces probabilities for real question/answer
            scope.reuse_variables()
            Dg = discriminator(tf.concat(q_in,Gz)) #Produces probabilities for generated question/answer pair


        #These functions together define the optimization objective of the GAN.
        d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
        g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

        tvars = tf.trainable_variables() #Don't know how many variables this is
        print(len(tvars))

        #The below code is responsible for applying gradient descent to update the GAN.
        trainerD = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.lr_decay)
        trainerG = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.lr_decay)
        d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #Only update the weights for the discriminator network.
        g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #Only update the weights for the generator network.

        update_D = trainerD.apply_gradients(d_grads)
        update_G = trainerG.apply_gradients(g_grads)
        #FINISHED DEFINING TensorFlow Graph
        #now cn train model with annoyingbrother.train(data)

    #TRAINING
    def train(self,data):

        batch_size = self.batch_size #Size of image batch to apply at each iteration.
        epochs = self.epochs
        iterations = len(data)/batch_size
        sample_directory = './figs' #Directory to save sample images from generator in.
        model_directory = './models' #Directory to save trained model to.

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:  
            sess.run(init)
            for _ in range(epochs):
                index = 0
                for i in range(iterations):
                    #grab question from data
                    qs = data[index:index+batch_size]
                    #grab real answer
                    answer = RealAnswer(qs)
                    index += batch_size

                    zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
                    xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
                    xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
                    xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
                    _,dLoss = sess.run([update_D,d_loss],feed_dict={q_in:qs,real_in:xs}) #Update the discriminator
                    _,gLoss = sess.run([update_G,g_loss],feed_dict={q_in:qs}) #update generator

                    if i % 100 == 0:
                        print("Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss))
                        z2 = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate another z batch
                        newZ = sess.run(Gz,feed_dict={z_in:z2}) #Use new z to get sample images from generator.
                        if not os.path.exists(sample_directory):
                            os.makedirs(sample_directory)
                        #Save sample generator images for viewing training progress.
                        save_images(np.reshape(newZ[0:36],[36,32,32]),[6,6],sample_directory+'/fig'+str(i)+'.png')
                    if i % 1000 == 0 and i != 0:
                        if not os.path.exists(model_directory):
                            os.makedirs(model_directory)
                        saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
                        print("Saved Model")



    def discriminator(self, d_in, reuse=False):
        #*original d_in = tf.placeholder(tf.float32, [None, 2*seq_length])
        #Have to set up reuse because we initialize this thing twise in the graph creating
        
        #Lets change this to be embedding LSTM probability model
        #a la https://gist.github.com/monikkinom/e97d518fe02a79177b081c028a83ec1c

        with tf.variable_scope("D_LSTM"):
            num_hidden = self.hidden_size
            cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

            val, _ = tf.nn.dynamic_rnn(cell, d_in, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])

            last = tf.gather(val, int(val.get_shape()[0]) - 1)

            #d_output_vocab = 1 (high probability means it looks like real data)
            weight = tf.Variable(tf.truncated_normal([num_hidden, self.d_output_vocab]))
            bias = tf.Variable(tf.constant(0.1, shape=[self.d_output_vocab]))

        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        
        return prediction

    def generator(self, q):

        #Lets change this to be embedding rnn model a la
        #https://github.com/hans/ipython-notebooks/blob/master/tf/TF%20tutorial.ipynb

        enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                                  name="inp%i" % t)
                   for t in range(self.seq_length)]

        labels = [tf.placeholder(tf.int32, shape=(None,),
                                name="labels%i" % t)
                  for t in range(self.seq_length)]

        weights = [tf.ones_like(labels_t, dtype=tf.float32)
                   for labels_t in labels]

        # Decoder input: prepend some "GO" token and drop the final
        # token of the encoder input
        dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
                   + enc_inp[:-1])

        # Initial memory value for recurrence.
        prev_mem = tf.zeros((self.batch_size, self.memory_dim))

        cell = rnn_cell.GRUCell(self.memory_dim)

        g_out, dec_memory = seq2seq.embedding_rnn_seq2seq(
            enc_inp, dec_inp, cell, self.input_vocab_size, self.output_vocab_size)
        
        return g_out

    def RealAnswer(self, q_in):
        #Contains Logic to make the real answer from the question
        #for now, just adding 2 to every word (annoying right?)
        key = 2*np.ones_like(q_in)
        real_answer = q_in + key
        return real_answer

