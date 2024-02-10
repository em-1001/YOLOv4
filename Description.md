# utils.py
Using **[..., 0:1]** instead of **[..., 0]** in PyTorch is a way to ensure that the result maintains a specific shape. Let me explain:

1. **Maintaining Dimension:**  
When you use **[..., 0:1]**, it selects the values along the last dimension (similar to **[..., 0]**), but it keeps the dimensionality intact. The resulting tensor will have an additional dimension compared to using **[..., 0]**.

2. **Broadcasting:**  
In many PyTorch operations, broadcasting is used to perform operations on tensors with different shapes. Adding a singleton dimension (e.g., using **[..., 0:1]**) can be useful to align dimensions for broadcasting.

3. **Consistency:**  
Using **[..., 0:1]** can be a design choice for consistency in handling dimensions. It makes sure that even if the selected range is a single value (width or height), the result is still a 2D tensor.

4. **Compatibility:**  
Some operations in PyTorch might expect a certain dimensionality for consistent processing. Using **[..., 0:1]** ensures that the tensor shape is consistent across different scenarios.

```py
import torch

# Example tensor
tensor_example = torch.tensor([[1, 2, 3],
                               [4, 5, 6]])

# Using [..., 0:1] to keep dimensionality intact
result_with_0_1 = tensor_example[..., 0:1]

# Using [..., 0] without keeping dimensionality
result_with_0 = tensor_example[..., 0]

print("Result with [..., 0:1]:")
print(result_with_0_1)
print("Shape:", result_with_0_1.shape)

print("\nResult with [..., 0]:")
print(result_with_0)
print("Shape:", result_with_0.shape)
```

### result

```py
Result with [..., 0:1]:
tensor([[1],
        [4]])
Shape: torch.Size([2, 1])

Result with [..., 0]:
tensor([1, 4])
Shape: torch.Size([2])
```

## iou_width_height
 iou_width_height function is designed to handle batched input, where each input tensor (boxes1 and boxes2) can represent multiple bounding boxes.

 In PyTorch, when you perform element-wise operations on tensors with broadcasting, the operation is automatically applied to all corresponding elements of the tensors. Broadcasting allows operations to be performed on tensors of different shapes, and PyTorch automatically adjusts the dimensions to make the operation valid.

In the context of the iou_width_height function:
- If boxes1 and boxes2 are single bounding boxes (i.e., 1D tensors of shape [2]), the operation will be applied element-wise, and you will get a single IoU value.

- If boxes1 and boxes2 are batched bounding boxes (i.e., 2D tensors of shape [batch_size, 2]), the operation will be applied independently for each pair of bounding boxes, and you will get a batch of IoU values.

Here's an example to illustrate:


```py
# iou_width_height Example
import torch

# Example values for box[2:4] and anchors
box_dimensions = torch.tensor([20.0, 30.0])  # Assuming width and height of the box
anchors = torch.tensor([[10.0, 20.0], [25.0, 35.0], [15.0, 25.0]])  # Example set of anchors

# Calculate IoU using the provided function
iou_result = iou_width_height(box_dimensions, anchors)

# Print the result
print("IoU Result:")
print(iou_result)
```

### result

```py
IoU Result:
tensor([0.3333, 0.6857, 0.6250])
```

## IoU, GIoU, DIoU, CIoU Loss Example
```py
# Assuming your IoU function is defined here...

box_preds_iou = torch.tensor([[2.0, 2.0, 4.5, 7.0]])  # (x, y, width, height)
box_labels_iou = torch.tensor([[3.0, 3.0, 4.0, 8.0]])  # (x, y, width, height)

# Example 1: IoU Calculation
iou_result = intersection_over_union(box_preds_iou, box_labels_iou, box_format="midpoint", iou_mode="IoU")
print("IoU Loss :", 1- iou_result.item())

# Example 2: GIoU Calculation
giou_result = intersection_over_union(box_preds_iou, box_labels_iou, box_format="midpoint", iou_mode="GIoU")
print("GIoU Loss:", 1 - giou_result.item())

# Example 3: DIoU Calculation
diou_result = intersection_over_union(box_preds_iou, box_labels_iou, box_format="midpoint", iou_mode="DIoU")
print("DIoU Loss:", 1 - diou_result.item())

# Example 4: CIoU Calculation
ciou_result = intersection_over_union(box_preds_iou, box_labels_iou, box_format="midpoint", iou_mode="CIoU")
print("CIoU Loss:", 1- ciou_result.item())

# Example 4: SIoU Calculation
siou_result = intersection_over_union(box_preds_iou, box_labels_iou, box_format="midpoint", iou_mode="SIoU")
print("SIoU Loss:", 1- siou_result.item())
```

### result

```py
IoU Loss : 0.501474916934967
GIoU Loss: 0.5518950819969177
DIoU Loss: 0.5215124785900116
CIoU Loss: 0.5215561389923096
SIoU Loss: 0.5263195931911469
```

## cells_to_bboxes
### cell_indices
1. **torch.arange(S):**  
Creates a 1D tensor with values from 0 to S-1. This tensor represents indices along one dimension of the image.

2. **.repeat(predictions.shape[0], 3, S, 1):**  
Repeats the 1D tensor **S** times along the last dimension, **3** times along the second-to-last dimension, and **predictions.shape[0]** times along the first dimension. This effectively creates a tensor of shape **(predictions.shape[0], 3, S, S, 1)**.

3. **.unsqueeze(-1):**  
Adds a new dimension at the end of the tensor. The resulting shape is now **(predictions.shape[0], 3, S, S, 1, 1)**.

4. **.to(predictions.device):**  
Moves the tensor to the same device as the **predictions** tensor.
Now, **cell_indices** is a tensor that represents the indices of the cells in the image grid.

5. **x = 1 / S * (box_predictions[..., 0:1] + cell_indices):**  
Calculates the x-coordinate of the bounding boxes. It adds the predicted x-coordinate (normalized between 0 and 1) to the cell indices and scales the result by **1/S**. This effectively maps the x-coordinate prediction to the entire width of the image.

6. **y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)):**  
Calculates the y-coordinate of the bounding boxes. It adds the predicted y-coordinate (normalized between 0 and 1) to the transposed cell indices and scales the result by **1/S**. The transposition is done to match the dimensions correctly.

7. **w_h = 1 / S * box_predictions[..., 2:4]:**  
Calculates the width and height of the bounding boxes. It scales the predicted width and height (which are already normalized between 0 and 1) by **1/S**.


```py
# example of cell_indices
cell_indices = torch.tensor([[[[[0], [1], [2], [3]],
                               [[0], [1], [2], [3]],
                               [[0], [1], [2], [3]],
                               [[0], [1], [2], [3]]]]])


```

y-coordinate를 구할 때 permute를 하는 이유는 x-coordinate를 구할 때는 위 cell_indices배열 기준으로 0열부터 3열까지 x값이 0부터 3으로 커져야 하지만 y-coordinate를 구할 때는 반대로 0행부터 3행까지 y값이 0부터 3으로 커져야 하기 때문에 permute를 통해 행과 열을 전치시켜주는 것이다.

## get_evaluation_bboxes
**boxes_scale_i** has the shape **(BATCH_SIZE, num_anchors * S * S, 6)**, so each element in the loop represents a tensor of shape **(num_anchors * S * S, 6)**.

```py
# Assuming boxes_scale_i has the shape (BATCH_SIZE, num_anchors * S * S, 6)
boxes_scale_i = torch.tensor([
    # ... Example values with shape (num_anchors * S * S, 6)
])

# Initialize an empty list to store bounding boxes for each image in the batch
bboxes = [[] for _ in range(boxes_scale_i.shape[0])]

# Iterate over predicted bounding boxes and append them to the corresponding image's list
for idx, (box) in enumerate(boxes_scale_i):
    bboxes[idx] += box
```

In this context:

- **boxes_scale_i.shape[0]** is **BATCH_SIZE**, representing the number of images in the batch.
- Each **box** during the loop iteration is a tensor of shape **(num_anchors * S * S, 6)** representing bounding boxes for a specific image.

After the loop, **bboxes** will be a list of lists, where each inner list corresponds to a set of bounding boxes for a specific image:

```py
bboxes = [
    # Image 1 bounding boxes with shape (num_anchors * S * S, 6)
    # Image 2 bounding boxes with shape (num_anchors * S * S, 6)
    # ... and so on for each image in the batch
]
```

Up to the above code is the result of one scale, and since it is repeated (three times) for all scales, the total number of bounding boxes per image in the **bboxes** becomes **(3 * num_anchors * S * S, 6)**, and only significant boxes are left by applying NMS to these boxes.

# loss.py
## Implementation of $𝟙^{obj}_ {i j}$ and $𝟙^{noobj}_{i j}$


In the expression **target[..., 1:5][obj]**, the **[obj]** at the end is used for boolean indexing or masking **($𝟙^{obj}_ {i j}$, $𝟙^{noobj}_{i j}$)** . Let's break down the expression to understand it better:

1. **target[..., 1:5]**: This part selects a subarray of the target tensor. Specifically, it selects the elements along the last dimension (index 1 to 4) of the tensor. This is often used in object detection models to extract the predicted bounding box coordinates.

2. **target[..., 0] == 1**: This creates a boolean mask by checking whether the values in the first channel of the last dimension of the **target** tensor are equal to 1. This typically represents the confidence scores for the presence of an object.

3. **target[..., 1:5][obj]**: Finally, the boolean mask **[obj]** is applied to the selected subarray. This means that only the elements corresponding to **True** values in the boolean mask will be returned. In the context of object detection, this operation is likely used to filter out predictions for which the confidence score is not equal to 1 (i.e., where an object is confidently detected).

```py
# Iobj_i, Inoobj_i Example

target = torch.tensor([[[[0, 0.2, 0.5, 0.7, 0.3, 0.9],
                        [1, 0.3, 0.4, 0.6, 0.2, 0.8]],
                       [[0, 0.5, 0.3, 0.6, 0.7, 0.2],
                        [-1, 0.4, 0.7, 0.5, 0.1, 0.6]]]])

obj = target[..., 0] == 1

print(target[..., 0])
print(obj)
print(target[..., 1:5][obj])
print(target[..., 1:5][target[..., 0] == 1]) # Equivalent expressions
```

### result
```py
tensor([[[ 0.,  1.],
         [ 0., -1.]]])
tensor([[[False,  True],
         [False, False]]])
tensor([[0.3000, 0.4000, 0.6000, 0.2000]])
tensor([[0.3000, 0.4000, 0.6000, 0.2000]])
```
