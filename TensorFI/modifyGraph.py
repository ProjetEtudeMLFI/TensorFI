# ModifyGraph - has the functions to instrument and modify the TensorFlow graph

import logging
from typing import List, Any, Dict

import numpy as np
import tensorflow as tf

from TensorFI import injectFault


def createFIFunc(
        operation,  # type: tf.Operation
        inputs,  # type: List[tf.Tensor]
        outputTypes,  # type:  List[tf.dtypes.DType]
        name,  # type: str
):  # type: (...) -> List[tf.Tensor]
    """Create a tensorflow operation representing a fault injection node"""
    # print "\nCreating FIfunc with ", opType, inputs, outputTypes, name

    fiFunc = None

    # Check the opType and return the corresponding function as Fifunc
    if operation.type == "Cast":
        # We have to special case Cast as it's expected to "remember" its type
        # This could be due to a bug in TensorFlow (at least it's not documented)
        fiFunc = injectFault.createInjectFaultCast(outputTypes[0])
    elif operation.type in injectFault.opTable:
        # Lookup the opTable and return the corresponding function (injectFault...)
        # This is the default case if there's an injectFault for the function
        fiFunc = injectFault.opTable[operation.type]
    else:
        # It's not a known operation, do not inject.
        logging.warning(
            "Operation {} should be either added to "
            "`excludeOps` or should have his injection implemented.".format(
                operation.type))
        if len(outputTypes) == 0:
            fiFunc = injectFault.opTable["Unknown"]
        else:
            # This is a temporary fix.
            return [tf.Tensor(operation, value_index=0, dtype=outputTypes[0])]

    # fiFunc should have been initialized (fiFunc != None)
    if fiFunc is None:
        raise ValueError("Unknown operation : " + str(operation.type))

    # Create a new TensorFlow operator with the corresponding fault injection function
    res = tf.numpy_function(fiFunc, inputs, outputTypes, name=name)
    # print "NewOp = ", res

    return res


# Done with createFIFunc


# Create fault injection equivalents for everything except {Placeholder, Variable, Constant, NoOp}
def excludeOps(op):
    # type: (tf.Operation) -> bool
    "Which operations to exclude from the instrumentation"
    result = (False
              or op.type in ("Placeholder", "Const", "NoOp", "VarHandleOp",
                             "VarIsInitializedOp", "ReadVariableOp",
                             "AssignVariableOp", "AssignAddVariableOp")
              or op.type.startswith("Variable"))
    return result


# Done with excludeOps


def modifyNodes(g, prefix):
    # type: (tf.Operation, str) -> Dict[tf.Tensor, tf.Tensor]
    """Insert nodes in the graph for fault injection corresponding to the original nodes"""
    ops = g.get_operations()  # type: List[tf.Operation]

    fiMap = {
    }  # Keeps track of the mapping between the FI node inserted and the original ones

    # Iterate over all the nodes in the TensorFlow graph
    for op in ops:
        # print("Modifying: " + pg.getOperation(op) )
        # Gather all the inputs in a list, replacing them with those from fiMap
        inputs = []
        for input in op.inputs:
            if input in fiMap:
                input = fiMap[input]
            inputs.append(input)

        # important attributes (e.g., strides) will be used as as the input of the tensor as well
        # Please check if the Op you want to inject has these attributes, if any, you should provide these to the customized Op for injection as well
        if (op.type == "Conv2D"):
            inputs.append((op.node_def.attr['strides'].list.i[:]))
            inputs.append(str(op.node_def.attr['padding'].s))
        elif (op.type == "LRN"):
            inputs.append(float(op.node_def.attr['bias'].f))
            inputs.append(float(op.node_def.attr['alpha'].f))
            inputs.append(float(op.node_def.attr['beta'].f))
        elif (op.type == "MaxPool"):
            inputs.append(np.asarray(op.node_def.attr['ksize'].list.i[:]))
            inputs.append(np.asarray(op.node_def.attr['strides'].list.i[:]))
            inputs.append(str(op.node_def.attr['padding'].s))

        # Create a new operation by wrapping debugPrint function and setting its inputs to op.inputs
        # Do this while preserving the control dependendencies of the original
        with g.control_dependencies(op.control_inputs):
            name = prefix + op.name

            # If operation is not one of the excluded operations for fault injections
            if not excludeOps(op):

                # Find the output types of all the outputs of op
                outputTypeList = []
                for output in op.outputs:
                    outputTypeList.append(output.dtype)

                # Create a new fault injection operation with the same inputs and outputs
                newOp = createFIFunc(op, inputs, outputTypeList, name)

                # Add newOp's output to the fiMap hashtable for each output of the current node
                # This will be used later to replace downstream operations that depend on it
                for i in range(0, len(op.outputs)):
                    output = op.outputs[i]
                    fiMap[output] = newOp[i]
                # Done with inner loop
            # Done if
        # Done with
    # Done with the outerloop loop
    return fiMap


# Done with modifyNodes
