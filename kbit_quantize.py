
# input: a tensor
# output: a tensor, result after quantizing values in input into 2^bits values.
def kbit_quantize(input):
  bits = 1
  t = 1/(2**(bits-1))
  thresh = []
  for i in range(1-2**(bits-1), 2**(bits-1)):
    thresh.append(t*i)
  print(thresh)

  quantizations = []
  for i in range(0-2**(bits-1), 0):
    quantizations.append(t*i)
  for i in range(1, 2**(bits-1)+1):
    quantizations.append(t*i)
  print(quantizations)
  masks = [tf.math.less(input, thresh[0])]
  for i in range(1, len(thresh)):
    masks.append(tf.math.greater_equal(input, thresh[i-1]) & tf.math.less(input, thresh[i]))
  masks.append(tf.math.greater_equal(input, thresh[len(thresh)-1]))

  output = input
  for i in range(0, len(masks)):
    output = tf.where(masks[i], tf.ones_like(input) * quantizations[i], output)

  return output

# input: a tensor
# output: a tensor, result after quantizing values in input into 2^bits values, plus zero.
def kbit_quantize_with_zero(input):
  bits = 4
  t = 1/(2**bits)
  thresh = []
  for i in range(1-2**bits, 2**bits, 2):
    thresh.append(t*i)

  quantizations = []
  q = 1/(2**(bits-1))
  for i in range((2**(bits-1))*-1, 2**(bits-1)+1):
    quantizations.append(q*i)

  masks = [tf.math.less(input, thresh[0])]
  for i in range(1, len(thresh)):
    masks.append(tf.math.greater_equal(input, thresh[i-1]) & tf.math.less(input, thresh[i]))
  masks.append(tf.math.greater_equal(input, thresh[len(thresh)-1]))

  output = input
  for i in range(0, len(masks)):
    output = tf.where(masks[i], tf.ones_like(input) * quantizations[i], output)

  return output

def ksign(input):
  output = tf.where(tf.math.greater_equal(input, 0), tf.ones_like(input), input)
  output = tf.where(tf.math.less(input, 0), tf.ones_like(input) * -1, output)
  return output
