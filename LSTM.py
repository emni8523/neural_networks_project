from ParagonFull import*

# First approach - use a basic LSTM to predict filter words. For each word, assign 0 if removed between transcript a and transcript c, 1 if kept
class LSTMModel():
    # segmented - if true, have punctuation marks like , . ? as there own tokens
    # lower - lower case
    def __init__(self, train=None, val=None, test=None, segmented=True, lower=True, model_type="lstm"):

        # load in data 
        X_train = [x.lower().strip() for x in list(train["transcript_a"])] if lower else [x.strip() for x in list(train["transcript_a"])]
        y_train = [x.lower().strip() for x in list(train["transcript_c"])] if lower else [x.strip() for x in list(train["transcript_c"])]

        X_val = [x.lower().strip() for x in list(val["transcript_a"])] if lower else [x.strip() for x in list(val["transcript_a"])]
        y_val = [x.lower().strip() for x in list(val["transcript_c"])] if lower else [x.strip() for x in list(val["transcript_c"])]

        X_test = [x.lower().strip() for x in list(test["transcript_a"])] if lower else [x.strip() for x in list(test["transcript_a"])]
        y_test = [x.lower().strip() for x in list(test["transcript_c"])] if lower else [x.strip() for x in list(test["transcript_c"])]

        # pre_process if segmented 
        if segmented:
            X_train = [self.pre_process_text(x) for x in X_train]
            X_test = [self.pre_process_text(x) for x in X_test]
            X_val = [self.pre_process_text(x) for x in X_val]
            y_train = [self.pre_process_text(x) for x in y_train]
            y_test = [self.pre_process_text(x) for x in y_test]
            y_val = [self.pre_process_text(x) for x in y_val]

        # tokenizer does not include , . ? in the filters if segmented is true, then it filters everything else out
        self.tokenizer = Tokenizer(oov_token="oov", filters='"#$%&()*+-/<=>@[\\]^_`{|}~\t\n') if segmented else Tokenizer(oov_token="oov")
        self.tokenizer.fit_on_texts(X_train + y_train + X_val + y_val + X_test + y_test)
        self.vocab_size = len(self.tokenizer.word_index)+1 #+1 for oov

        # about 90% of the data is less than 40 words long
        self.max_length = 40

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.segmented = segmented
        self.lower = True
        self.model_type = model_type


    # use re to add a space before notable punctuation
    def pre_process_text(self, text):
        text = re.sub(r'([,.?!:;])',r' \1',text)
        return text


    # function that takes in raw text and clean text; returns tokens of raw_text and a list of binary labels (1 for remove, 2 for keep)
    def make_binary_labels(self, raw, clean, return_tokens=True):
        raw_words = raw.split()
        clean_words = clean.split()
        
        matcher = difflib.SequenceMatcher(None, raw_words, clean_words) #isJunk = None to ignore no items
        labels = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                labels += [2] * (i2 - i1)   # keep
            else:
                labels += [1] * (i2 - i1)   # remove
        
        # for model 2, which uses different encodings
        if not return_tokens:
            return raw_words, labels

        return self.tokenizer.texts_to_sequences([raw])[0], labels
    

    # prints a histogram of the number of words per sequence
    def plot_seq_lengths(self):
        X_train_lengths = [len(i.split()) for i in self.X_train]
        print(f"First few lengths: {X_train_lengths[:5]}")

        plt.title("Words per Sequence")
        plt.xlabel("Number of Words")
        plt.hist(X_train_lengths)
        plt.plot()


    # take in all raw/clean data and reformat as raw_tokens/clean labels after padding
    def encode_example(self, x, y):
        raw_token, raw_label = self.make_binary_labels(x, y)

        # pad sequences to each be the same length
        tokens_padded = pad_sequences([raw_token], maxlen=self.max_length, padding='post')[0]
        labels_padded = pad_sequences([raw_label], maxlen=self.max_length, padding='post')[0]

        return tokens_padded, labels_padded
    

    # create a binary dataset from raw->clean dataset
    def create_binary_dataset(self,X,Y):
        x_toks = []
        y_labels = []

        for x,y in zip(X,Y):
            toks, labels = self.encode_example(x,y)
            x_toks.append(toks)
            y_labels.append(labels)
        
        return np.array(x_toks), np.array(y_labels)

    
    # print the proportions of each label to check if padding makes sense 
    def print_props(self):
        unique, counts = np.unique(self.y_train_labels,return_counts=True)
        for u, c in zip(unique, counts):
            print(f"Percentage {u} is {c / counts.sum()}")

    
    # train the model
    def train(self,epochs):
        self.X_train_tokens, self.y_train_labels = self.create_binary_dataset(self.X_train, self.y_train)
        self.X_val_tokens, self.y_val_labels = self.create_binary_dataset(self.X_val, self.y_val)
        self.X_test_tokens, self.y_test_labels = self.create_binary_dataset(self.X_test, self.y_test)
        
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, input_shape=(self.vocab_size,), output_dim=32, mask_zero=True), # masks zero so we can use sparsecategoricalcrossentropy and accuracy without the padding affecting the loss
            Bidirectional(LSTM(units=100, dropout=0.2, return_sequences=True)),
            Bidirectional(LSTM(units=50, dropout=0.2, return_sequences=True)),
            Dense(3, activation='softmax') # pad, keep, remove
        ])
        self.model.summary()

        # add sample weights to emphasize getting the removed tokens right more frequently
        sample_weights = np.ones_like(self.y_train_labels) # same shape, all ones
        sample_weights[self.y_train_labels==0] = 0 # this also makes the padded words not count in the accuracy calculation
        sample_weights[self.y_train_labels==1] = 3
        sample_weights[self.y_train_labels==2] = 1

        self.model.compile(
            optimizer= Adam(0.001), # learning rate
            loss= SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()])

        callbacks = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=4, restore_best_weights=True) # implement early stopping to prevent overfitting
        history = self.model.fit(self.X_train_tokens,
                self.y_train_labels,
                epochs=epochs,
                validation_data=(self.X_val_tokens, self.y_val_labels),
                sample_weight=sample_weights,
                callbacks=callbacks)
        
        # plot loss and accuracy over epochs
        plt.plot(history.history['sparse_categorical_accuracy'], label = 'accuracy')
        plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'val_accuracy')
        plt.plot(history.history['loss'], label = 'loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')

        plt.title('Accuracy and Loss per Epoch')
        plt.xlabel('Epoch')
        plt.legend()

        plt.plot()


    # predict a raw text input, returning the predicted labels, the sentnece, and the tokenized sequence
    def predict_sequence(self,raw_text):
        # tokenize and pad
        seq = self.tokenizer.texts_to_sequences([self.pre_process_text(raw_text)] if self.segmented else [raw_text])
        seq = pad_sequences(seq, maxlen=self.max_length, padding='post')

        # predict sequence
        preds = self.model.predict(seq)[0]
        pred_classes = np.argmax(preds, axis=-1)

        # reverse word_index to get words from tokens
        reverse_index = {v: k for k, v in self.tokenizer.word_index.items()}
        reverse_index[0] = '<PAD>'

        # only keep words labelled 2
        pred_sentence = ''
        for idx, label in zip(seq[0],pred_classes):
            if idx == 0: #ignore padding
                continue 

            if label == 2: # only keep 2s
                word = reverse_index[idx]
                pred_sentence += word + ' '

        # put sentence back together
        pred_sentence = re.sub(r' ([,.?!:;])',r'\1',pred_sentence)

        return pred_classes, pred_sentence, seq


    # print a confusion matrix given tokens and labels
    def print_cm(self,toks,labels):  
        predictions = self.model.predict(toks)
        y_pred = np.argmax(predictions, axis=-1).flatten()
        y_gold = labels.flatten()

        # remove padding values 
        mask = (y_gold != 0)
        y_pred = y_pred[mask]
        y_gold = y_gold[mask]

        cm = confusion_matrix(y_gold, y_pred, labels=[1,2])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        labels = ['Remove', 'Keep']

        disp.ax_.set_title("Filter Model")
        disp.ax_.set_xticklabels(labels,rotation=90)
        disp.ax_.set_yticklabels(labels)

        # use sklearn metrics 
        print(f"F1 score: {f1_score(y_gold, y_pred)}")
        print(f"Classification Report: \n {classification_report(y_gold, y_pred)}")

        plt.show()


    # evaluate on test data, predict a given sentence, and print the cm
    def show_eval(self):
        loss, accuracy = self.model.evaluate(self.X_test_tokens, self.y_test_labels)

        print(f"Test loss: {loss}")
        print(f"Test accuracy: {accuracy}")

        print("\n-------------\n")

        print("Testing on the sentence: i, uh, don't, i don't think that is true")
        print("Expected: i don't think that is true")
        print(f"Result: {self.predict_sequence("i, uh, don't, i don't think that is true")[1]}")

        self.print_cm(self.X_test_tokens,self.y_test_labels)

        
    # train an evaluate
    def run(self, epochs):
        self.train(epochs)
        self.show_eval()