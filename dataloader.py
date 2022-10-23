
    def mapper():
        keys_tensor = tf.constant([b'A', b'T', b'G', b'C',b'a', b't', b'g', b'c'])
        vals_tensor = tf.constant([0,3,1,2,0,3,1,2])
        init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        table = tf.lookup.StaticHashTable(init, default_value=4)
        return table

    @tf.function
    def process_string(string, table=mapper(), onehot=True, label_onehot=True):
        x = tf.strings.split(string, sep=',')
        s = tf.strings.bytes_split(x[1]) 
        f=table.lookup(s) 
        f2 = tf.strings.to_number(x[0], tf.int32)
        f2=tf.cast(f2, dtype=tf.int32)
        if label_onehot:
            f2 = tf.one_hot(f2, depth=4)

        if onehot:
            return tf.one_hot(f, depth=4, dtype=tf.float16, on_value=0.99, off_value=0.0033), f2
        else:
            return f
