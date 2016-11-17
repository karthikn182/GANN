

#Things to-do
#1. make output of Generator a sequence (not probabilties!)
#2. Figure out concatenating in the right manner!
#3. make sure train() function is correct

import time
import numpy as np
import tensorflow as tf

class AnnoyingBrotherConfig(object):
    """Define HyperParameters"""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2 #Not Implemented: can be used to make Discriminator deeper (multi-LSTM cells)
    sequence_length = 5 
    hidden_size = 50 #size of one lstm cell
    memory_dim = 100 #memory dimension for GRU cell in seq2seq
    max_epoch = 2
    keep_prob = 0.5 #for dropout, Not Implemented
    lr_decay = 0.8
    batch_size = 1
    input_vocab_size = 10
    output_vocab_size = 12 #adding 2 to each word
    d_output_vocab = 1 #aka number of classes (in this case P(real data))

class ABModel(object):

    def __init__(self,config):
        print("initializing Annoying Brother Init model")
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
        self.keep_prob = config.keep_prob #for dropout if we decide

        self.batch_size = config.batch_size
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size
        self.d_output_vocab = 2
        #--------------D and G Models----------------------------#
        def discriminator(d_in, reuse=False):
            print("discriminator architecture")
            #An RNN classifier - dynamic, so input is (batch,seq,features)
            #*original d_in = tf.placeholder(tf.float32, [None, 2*seq_length,1])
            #Have to set up reuse because we initialize this thing twise in the graph creating

            #Lets change this to be embedding LSTM probability model
            #a la https://gist.github.com/monikkinom/e97d518fe02a79177b081c028a83ec1c
            #could also use a more complicated/ deeper classification model
            #Ok so dynamic rnn unrolls inputs automatically
            #aka takes in a single tensor(batch,seq_len,dim)

            num_hidden = self.hidden_size
            self.dcell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)


            # Have to cast d_in to float
            self.val, _ = tf.nn.dynamic_rnn(self.dcell, d_in,dtype="float32")
            self.val = tf.transpose(self.val, [1, 0, 2]) #transposes so we can get just the last time step

            self.last = tf.gather(self.val, int(self.val.get_shape()[0]) - 1) #get last output

            #make prediction based on last output
            #d_output_vocab = 1 (high probability means it looks like real data)
            self.dweight = tf.Variable(tf.truncated_normal([num_hidden, self.d_output_vocab]))
            self.dbias = tf.Variable(tf.constant(0.1, shape=[self.d_output_vocab]))
            self.prediction = tf.nn.softmax(tf.matmul(self.last, self.dweight) + self.dbias)

            return self.prediction

        def generator(enc_inp):
            #print(len(enc_inp))
            #print(tf.shape_n(enc_inp))
            #this one still needs work!
            #input is encoder_input (a list of inputs)
            #output is ?argmax? of decoder_output

            print("generator architecture")

            #Lets change this to be embedding rnn model a la
            #https://github.com/hans/ipython-notebooks/blob/master/tf/TF%20tutorial.ipynb

            #self.labels = [tf.placeholder(tf.int32, shape=(None,),
             #                       name="labels%i" % t)
              #        for t in range(self.seq_length)]

            #self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
             #          for labels_t in self.labels]

            # Decoder input: prepend some "GO" token and drop the final
            # token of the encoder input
            self.dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
                       + enc_inp[:-1])

            # Initial memory value for recurrence.
            self.prev_mem = tf.zeros((self.batch_size, self.memory_dim))

            self.cell = tf.nn.rnn_cell.GRUCell(self.memory_dim)
            '''
            embedding_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              num_encoder_symbols,
                              num_decoder_symbols,
                              embedding_size,
                              output_projection=None,
                              feed_previous=False,
                              dtype=None,
                              scope=None):
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py
                              '''      
            #d_out is a list of tensors
            self.d_out, self.dec_memory = tf.nn.seq2seq.embedding_rnn_seq2seq(
                self.enc_inp, self.dec_inp, self.cell, self.input_vocab_size, self.output_vocab_size,self.memory_dim)

            #might need to take softargmax? AND pack it back into one tensor!
            self.g_out = self.pack_sequence(self.d_out)
            print("Generator out shape:",self.g_out.get_shape())
            return self.g_out

        #INITIALIZING GRAPH
        tf.reset_default_graph()

        #This initializer is used to initialize all the weights of the network (not done until train() is called)
        initializer = tf.truncated_normal_initializer(stddev=self.init_scale)

        #Define Generator variables
        print("generator vars")
        with tf.variable_scope("G") as scope:
            #q_in = tf.placeholder(shape=[None,self.seq_length,1],dtype=tf.int32) #Question vector (batching,seq_length,word_dimension)

            #Generator input placeholder: encoder_inputs are a list of tensors. len(list) = eq length
            
            self.enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
                        for t in range(self.seq_length)]
            
            self.Gz = generator(self.enc_inp) #Generates answer from question input

        #Only create the Discriminator Variables once!
        print("discriminator vars")
        with tf.variable_scope("D") as scope:
            #Discriminator input placeholder (none,2*seq_length,1)
            self.real_in = tf.placeholder(shape=[None,self.seq_length,self.output_vocab_size],dtype=tf.float32) #Real "Question/Answer" placeholder 
            self.Dx = discriminator(self.real_in) #Produces probabilities for real question/answer
            print("now trying to input G_out into Discriminator")
            scope.reuse_variables()

            #how to combine question and G_out? Input concatenation of qd,Gz
            self.Dg = discriminator(self.Gz) #Produces probabilities for generated question/answer pair


        #These functions together define the optimization objective of the GAN.
        self.d_loss = -tf.reduce_mean(tf.log(self.Dx) + tf.log(1.-self.Dg)) #This optimizes the discriminator.
        self.g_loss = -tf.reduce_mean(tf.log(self.Dg)) #This optimizes the generator.



        #The below code is responsible for applying gradient descent to update the GAN.
        self.trainerD = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.lr_decay)
        self.trainerG = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.lr_decay)

        self.tvars = tf.trainable_variables() #Don't know how many variables this is
        print("tvars length: ",len(self.tvars)) #length 18

        print("about to comput gradients")
        self.d_grads = self.trainerD.compute_gradients(self.d_loss,self.tvars[9:]) #Only update the weights for the discriminator network.
        self.g_grads = self.trainerG.compute_gradients(self.g_loss,self.tvars[0:9])
        #print("computing d gradients",self.d_grads)
        #print("computing g gradients",self.g_grads)


        #d_params = [v for v in tvars if v.name.startswith('D')]
        #g_params = [v for v in tvars if v.name.startswith('G')]
        #d_grads = trainerD.compute_gradients(d_loss,d_params) #Only update the weights for the discriminator network.
        #g_grads = trainerG.compute_gradients(g_loss,g_params) #Only update the weights for the generator network.

        self.update_D = self.trainerD.apply_gradients(self.d_grads)
        self.update_G = self.trainerG.apply_gradients(self.g_grads)
        #FINISHED DEFINING TensorFlow Graph
        #now cn train model with annoyingbrother.train(data)


    #TRAINING
    def train(self,data):
        print("training")

        batch_size = self.batch_size #Size of image batch to apply at each iteration.
        epochs = self.max_epoch
        iterations = len(data)//batch_size
        sample_directory = './figs' #Directory to save sample images from generator in.
        model_directory = './models' #Directory to save trained model to.

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:  
            sess.run(init)
            for _ in range(epochs):
                index = 0
                for i in range(iterations):
                    #grab integer question from data for generator
                    qg = data[index:index+batch_size]
                    print("qg",qg)
                    index += batch_size

                    #grab real answer from question, resized for discriminator (batch,seq_length,features)
                    answer = self.RealAnswer(qg)
                    print("answer",answer)

                    '''
                    Just keeping this in here for now...
                    Code from DCGANN Tutorial their feed dict would look like
                    zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) #Generate a random z batch
                    xs,_ = mnist.train.next_batch(batch_size) #Draw a sample batch from MNIST dataset.
                    xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
                    xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
                    '''
                    #feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
                    #sorry hardcoding the batch size one here  :/
                    gfeed_dict = {self.enc_inp[t]: [qg[0][t]] for t in range(self.seq_length)}
                    dfeed_dict = {self.enc_inp[t]: [qg[0][t]] for t in range(self.seq_length)}
                    dfeed_dict.update({self.real_in: answer})


                    _,dLoss = sess.run([self.update_D,self.d_loss],dfeed_dict)#Update the discriminator
                    _,gLoss = sess.run([self.update_G,self.g_loss],gfeed_dict) #update generator

                    if i % 100 == 0:
                        #print a question/Answer
                        print("Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss))
                        q2 = data[0:0+batch_size] #using the first batch of question to see how we improve
                        newA = sess.run(self.Gz,feed_dict={self.enc_inp: [q2[0][t]] for t in range(self.seq_length)})

                        if not os.path.exists(sample_directory):
                            os.makedirs(sample_directory)
                        #Save sample generator images for viewing training progress.
                        save_Answer(q2,newA,sample_directory+'/fig'+str(i))
                    if i % 1000 == 0 and i != 0:
                        if not os.path.exists(model_directory):
                            os.makedirs(model_directory)
                        saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
                        print("Saved Model")



    def RealAnswer(self, q_in):
        #Contains Logic to make the real answer from the question
        #for now, just adding 2 to every word (annoying right?)
        #a one hot answer!
        key = 2*np.ones_like(q_in)
        real_answer = q_in + key
        return self.convert_onehot(real_answer,self.output_vocab_size)

    def convert_onehot(self, a,classes):
        z = (np.arange(classes) == a[:,:,None]-1).astype(float)
        #z = np.zeros(list(a.shape) + [classes])
        #print(z)
        #z[list(np.indices(z.shape[:-1])) + [a]] = 1
        return z

        def save_answer(self, q, a, path):
            answer = np.concatenate((q,a),axis=1)
            return np.savetxt(path,answer)


    #Extremely useful for transfering from seq2seq stuff to the dynamic rnn classification
    #https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
    def unpack_sequence(self, tensor):
        """Split the single tensor of a sequence into a list of frames."""
        #return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))
        return tf.unpack(tensor,axis=1)


    def pack_sequence(self,sequence):
        """Combine a list of the frames into a single tensor of the sequence."""
        #return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])
        return tf.pack(sequence,axis=1)

