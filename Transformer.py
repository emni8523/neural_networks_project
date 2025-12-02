from ParagonFull import*
from LSTM import LSTMModel

# need this wrapper class to avoid getting errors, bert takes in only tensorflow tensors and not keras tensors
@register_keras_serializable(package="Custom")
class BertLayer(Layer):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs) # init Layer class
        self.model_name = model
        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True)
        self.bert = TFBertModel.from_pretrained( #bert model
            model,
            config=self.config,
            from_pt=True
        )

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.hidden_states
        layers = hidden_states[-4:] # get last 4 hidden layers
        x = tf.concat(layers, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        seq = input_shape[0][1]
        return (batch, seq, self.config.hidden_size * 4)
    
    def get_config(self):
        config = super().get_config()
        config.update({"model": self.model_name})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)




# Transformer model, inherited from bart
class TransformerModel(LSTMModel):
    def __init__(self, train=None, val=None, test=None, batch=8, segmented=True, lower=True, model_type="transformer"):
        super().__init__(train=train, val=val, test=test, segmented=segmented, lower=lower, model_type=model_type) # init from parent class

        # different tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.batch = batch
    

    # when assigning labels, BERT will separate words like testing into test and ing. Use word_ids here to ensure that each subword has the same class as other subwords
    def subword_segment(self, word_ids, y_labels):
        labels = []

        for i in word_ids:
            if i is None:
                labels.append(0) # label special characters the same as padding
            else:
                labels.append(y_labels[i])
        
        return labels
    

    # take in clean and raw text, and return an encoding dictionary of tensors that includes tokens ('input_ids'), labels ('labels'), and attention_mask (used during training, 1 for use a word, 0 for not (special character, padding, etc.))
    def encode_example(self, X, y):
        X_raw_words, y_labels = self.make_binary_labels(X,y,return_tokens=False) #return raw words instead
        
        encoding = self.tokenizer.encode_plus(X_raw_words, 
                                        max_length=self.max_length, #est earlier
                                        truncation=True, 
                                        padding="max_length",
                                        is_split_into_words=True,
                                        return_token_type_ids=False, #don't need these
                                        return_tensors='tf' #for tensorflow
                                        )
        
        #create labels to give subwords the same class using word ids
        word_ids = encoding.word_ids()
        labels = self.subword_segment(word_ids, y_labels)
        
        encoding["labels"] = labels
        encoding["input_ids"] = encoding["input_ids"][0]
        encoding["attention_mask"] = encoding["attention_mask"][0]
        return encoding
        
    
    # training method    
    def train(self, epochs):
        train_encodings = [self.encode_example(x,y) for x,y in zip(self.X_train, self.y_train)]
        test_encodings = [self.encode_example(x,y) for x,y in zip(self.X_test, self.y_test)]
        val_encodings = [self.encode_example(x,y) for x,y in zip(self.X_val, self.y_val)]

        self.train_ds = Dataset.from_list(train_encodings)
        self.test_ds = Dataset.from_list(test_encodings)
        self.val_ds = Dataset.from_list(val_encodings)

        # use tokens and attention mask as the inputs
        input_ids = Input(shape=(self.max_length,), name="input_ids", dtype=tf.int32)
        attention_mask = Input(shape=(self.max_length,), name="attention_mask", dtype=tf.int32)

        # return the final 4 hidden layers from Bert, using uncased or cased to match with training data
        sequence_outputs = BertLayer(model="bert-base-uncased" if self.lower else "bert-base-cased") ((input_ids, attention_mask))

        x = Dropout(0.3)(sequence_outputs)
        x = TimeDistributed(Dense(256, activation='relu'))(x) # time distributed due to attention
        x = Dropout(0.3)(x)
        #x = TimeDistributed(Dense(64, activation='relu'))(x)
        #x = Dropout(0.2)(x)
        y = TimeDistributed(Dense(3, activation='softmax'))(x) #output

        self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=y)
        self.model.summary()

        # sample weights
        sample_weights = np.ones_like(self.train_ds["labels"]) 
        sample_weights[self.train_ds["labels"]==0] = 0
        sample_weights[self.train_ds["labels"]==1] = 3
        sample_weights[self.train_ds["labels"]==2] = 1

        # train the model
        self.model.compile(
            optimizer= Adam(3e-5), # learning rate
            loss= SparseCategoricalCrossentropy(),
            metrics= [SparseCategoricalAccuracy()])


        self.train_tf = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": np.array(self.train_ds["input_ids"], dtype=np.int32),
                "attention_mask": np.array(self.train_ds["attention_mask"], dtype=np.int32)
            },
            np.array(self.train_ds["labels"]),
            sample_weights.astype(np.float32))) #sample_weights added here

        self.val_tf = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": np.array(self.val_ds["input_ids"], dtype=np.int32),
                "attention_mask": np.array(self.val_ds["attention_mask"], dtype=np.int32)
            },
            np.array(self.val_ds["labels"])))

        callbacks = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, restore_best_weights=True) # implement early stopping to prevent overfitting
        history = self.model.fit(self.train_tf.batch(self.batch), 
                            epochs=epochs, 
                            validation_data=self.val_tf.batch(self.batch),
                            callbacks=callbacks
                            )

        # plot loss and accuracy over epochs
        plt.plot(history.history['sparse_categorical_accuracy'], label = 'accuracy')
        plt.plot(history.history['loss'], label = 'loss')
        plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'val_accuracy')
        plt.plot(history.history['val_loss'], label = 'val_loss')

        plt.xlabel('Epoch')
        plt.legend()

        plt.plot()


    # predict a raw text input
    def predict_sequence(self,raw_text):
        # tokenize and pad
        encoding = self.tokenizer.encode_plus(raw_text.split(), 
                                        max_length=self.max_length, #est earlier
                                        truncation=True, 
                                        padding="max_length",
                                        is_split_into_words=True,
                                        return_token_type_ids=False, #don't need these
                                        return_tensors='tf' #for tensorflow
                                        )

        # predict sequence
        preds = self.model.predict({"input_ids": encoding["input_ids"],
                            "attention_mask": encoding["attention_mask"]})
        
        pred_classes = np.argmax(preds, axis=-1)[0]
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        word_ids = encoding.word_ids(batch_index=0)

        pred_words = []
        current_word_tokens = [] # keep track of current word vs word_id for the sake of subwords
        current_word_id = None
        for tok, label, wid in zip(tokens,pred_classes,word_ids):
            if wid is None: # for special tags
                continue

            if wid != current_word_id: # new word
                if current_word_id is not None and first_label == 2:
                    pred_words.append("".join(current_word_tokens))

                current_word_id = wid
                current_word_tokens = [tok.replace("##", "")]
                first_label = label
                
            else: # for subwords
                current_word_tokens.append(tok.replace("##",""))
            
        if current_word_id is not None and first_label == 2:
            pred_words.append("".join(current_word_tokens))

        # put sentence back together
        pred_sentence = " ".join(pred_words)
        pred_sentence = re.sub(r' ([,.?!:;])',r'\1',pred_sentence)
        pred_sentence = re.sub(r" ' ", r"'", pred_sentence)
        
        return pred_classes, pred_sentence, encoding

    # print the confusion matrix 
    def print_cm(self, tens):
        predictions = self.model.predict(tens.batch(self.batch))
        y_pred = np.argmax(predictions, axis=-1).flatten()

        # get gold values across batches and flatten
        y_list = []
        for batch in tens:
            _, y_batch = batch   
            y_list.append(y_batch.numpy()) 
        y = np.concatenate(y_list, axis=0) 

        # mask padded labels
        mask = (y != 0)
        y_pred = y_pred[mask]
        y = y[mask]

        cm = confusion_matrix(y, y_pred, labels=[1,2])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        labels = ['Remove', 'Keep']

        disp.ax_.set_title("Filter Model")
        disp.ax_.set_xticklabels(labels,rotation=90)
        disp.ax_.set_yticklabels(labels)

        # use sklearn metrics 
        print(f"F1 score: {f1_score(y, y_pred)}")
        print(f"Classification Report: \n {classification_report(y, y_pred)}")

        plt.show()


    # evaluate, test on example, and print confusion matrix
    def show_eval(self):
        test_tf = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": np.array(self.test_ds["input_ids"], dtype=np.int32),
                "attention_mask": np.array(self.test_ds["attention_mask"], dtype=np.int32)
            },
            np.array(self.test_ds["labels"])
        ))

        loss, accuracy = self.model.evaluate(test_tf.batch(self.batch))

        print(f"Test loss: {loss}")
        print(f"Test accuracy: {accuracy}")

        print("\n-------------\n")

        print("Testing on the sentence: i, uh, don't, i don't think that is true")
        print("Expected: i don't think that is true")
        test = "i, uh, don't, i don't think that is true"
        print(f"Result: {self.predict_sequence(test)[1]}")

        self.print_cm(test_tf)