from typing import Tuple
from absl import app
from absl import flags
import numpy as np
import scipy.sparse
from scipy.sparse import base
import sklearn.metrics
import tensorflow as tf
from graph_clustering import dmon
from graph_clustering import gcn
from graph_clustering import utils
tf.compat.v1.enable_v2_behavior()
from graph_clustering import graph_construct
from scipy.sparse import csr_matrix


def convert_scipy_sparse_to_sparse_tensor(
    matrix):
  """Converts a sparse matrix and converts it to Tensorflow SparseTensor.
  Args:
    matrix: A scipy sparse matrix.
  Returns:
    A ternsorflow sparse matrix (rank-2 tensor).
  """
  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)



def build_dmon(input_features,
               input_graph,
               input_adjacency):
  """Builds a Deep Modularity Network (DMoN) model from the Keras inputs.
  Args:
    input_features: A dense [n, d] Keras input for the node features.
    input_graph: A sparse [n, n] Keras input for the normalized graph.
    input_adjacency: A sparse [n, n] Keras input for the graph adjacency.
  Returns:
    Built Keras DMoN model.
  """
  output = input_features
  for n_channels in [256]:   # FLAGS.architecture:
    output = gcn.GCN(n_channels)([output, input_graph])
  pool, pool_assignment = dmon.DMoN(
      n_clusters=20,
      collapse_regularization=1,
      dropout_rate=0.5)([output, input_adjacency])
  '''
  FLAGS.n_clusters,
      collapse_regularization=FLAGS.collapse_regularization,
      dropout_rate=FLAGS.dropout_rate)([output, input_adjacency]
  '''
  return tf.keras.Model(
      inputs=[input_features, input_graph, input_adjacency],
      outputs=[pool, pool_assignment])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Load and process the data (convert node features to dense, normalize the
  # graph, convert it to Tensorflow sparse tensor.
  adjacency, feature = graph_construct.load_graph()#原adj: sparse 现在: numpy
  features = tf.convert_to_tensor(feature)
  n_nodes = adjacency.shape[0]
  feature_size = features.shape[1]
  #graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  #np.asarray(x).astype('float32')
  adj = tf.constant(tf.convert_to_tensor(adjacency.astype('int32')))
  graph_idx = tf.where(tf.not_equal(adj, 0))
  graph = tf.SparseTensor(graph_idx, tf.gather_nd(adj, graph_idx), adj.get_shape())
  #graph = convert_scipy_sparse_to_sparse_tensor()
  adj2 = csr_matrix(adjacency)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(utils.normalize_graph(adj2))#to sparse tensr
  print(graph.dtype," ",features.dtype," ",graph_normalized.dtype)

  # Create model input placeholders of appropriate size
  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = build_dmon(input_features, input_graph, input_adjacency)

  # Computes the gradients wrt. the sum of losses, returns a list of them.
  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(0.0001)#learning rate
  model.compile(optimizer, None)

  for epoch in range(300):#FLAGS.n_epochs
    loss_values, grads = grad(model, [features, graph_normalized, graph])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, losses: ' +
          ' '.join([f'{loss_value.numpy():.4f}' for loss_value in loss_values]))

  # Obtain the cluster assignments.
  _, assignments = model([features, graph_normalized, graph], training=False)
  assignments = assignments.numpy()
  clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
  np.save('clusters',clusters)


if __name__ == '__main__':
  app.run(main)