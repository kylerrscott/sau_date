# input: a tensor
# output: a tensor, result after quantizing values in input into 2^bits values, plus zero.
def kbit_quantize(input):
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
