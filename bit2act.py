def bit2_act(input):
  thresh0 = 0
  thresh1 = 0.5
  thresh2 = -0.5

  mask1 = tf.math.greater_equal(input, thresh0) & tf.math.less(input, thresh1)
  mask2 = tf.math.greater_equal(input, thresh1)
  mask3 = tf.math.greater_equal(input, thresh2) & tf.math.less(input, thresh0)
  mask4 = tf.math.less(input, thresh2)
  
  output = tf.where(mask1, tf.ones_like(input) * 0.5, input)
  output = tf.where(mask2, tf.ones_like(input), output)
  output = tf.where(mask3, tf.ones_like(input) * -0.5, output)
  output = tf.where(mask4, tf.ones_like(input) * -1, output)

  return output
