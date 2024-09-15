import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    height, width, _ = image.shape
    gridX = np.arange(width, dtype=np.int16)
    gridY = np.arange(height, dtype=np.int16)
    vy, vx = np.meshgrid(gridX, gridY)
    grid_row, grid_col= vx.shape
    ctrl_pts = source_pts.shape[0]
    q_pts = source_pts.reshape(source_pts.shape[0], source_pts.shape[1], 1, 1)
    p_pts = target_pts.reshape(target_pts.shape[0], target_pts.shape[1], 1, 1)

    v = np.vstack((vx.reshape(1, grid_row, grid_col), vy.reshape(1, grid_row, grid_col)))  

    w = 1.0 / (np.sum((p_pts - v) ** 2, axis=1) + eps) ** alpha
    w /= np.sum(w, axis=0, keepdims=True) 
    p_star = np.zeros((2, grid_row, grid_col), np.float32)
    q_star = np.zeros((2, grid_row, grid_col), np.float32)
    for i in range(ctrl_pts):
        p_star += w[i] * p_pts[i]  
        q_star += w[i] * q_pts[i]   
    
    p_hat = p_pts - p_star
    q_hat = q_pts - q_star
    ortho_p_hat = p_hat[:, [1, 0], :, :]  
    ortho_p_hat[:,1,:,:] = -ortho_p_hat[:,1,:,:]
    mul_left = np.stack([p_hat, ortho_p_hat], axis=0) 
    mul_left = mul_left.transpose(1, 0, 2, 3, 4)

    vpstar = v - p_star
    ortho_vpstar = vpstar[[1, 0], :, :]
    ortho_vpstar[1,:,:] = -ortho_vpstar[1,:,:]
    mul_right = np.stack([vpstar, ortho_vpstar], axis=0)
    mul_right = mul_right.reshape(1, 2, target_pts.shape[1], grid_row, grid_col)
    mul_right = np.repeat(mul_right, ctrl_pts, axis=0)

    A = np.matmul((w.reshape(ctrl_pts, 1, 1, grid_row, grid_col) * mul_left).transpose(3, 4, 0, 1, 2), mul_right.transpose(3, 4, 0, 1, 2))
    reshaped_q_hat = q_hat.reshape(ctrl_pts, 1, 2, grid_row, grid_col)
    fr = np.sum(reshaped_q_hat.transpose(3, 4, 0, 1, 2) @ A, axis=2)
    fr = fr.reshape(grid_row, grid_col, 2).transpose(2, 0, 1)
    fr_norm = np.linalg.norm(fr, axis=0, keepdims=True)  
    normed_fr = fr / fr_norm

    vpstar_norm = np.linalg.norm(vpstar, axis=0, keepdims=True) 
    
    transform_map = np.zeros((2, grid_row, grid_col), dtype=np.int16)
    transform_map = vpstar_norm * normed_fr + q_star
    nan_mask = fr_norm[0] == 0
    nan_mask_flat = np.flatnonzero(nan_mask)
    nan_mask_anti_flat = np.flatnonzero(~nan_mask)
    transform_map[0][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transform_map[0][~nan_mask])
    transform_map[1][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transform_map[1][~nan_mask])

    transform_map[transform_map < 0] = 0
    transform_map[0][transform_map[0] > grid_row - 1] = 0
    transform_map[1][transform_map[1] > grid_col - 1] = 0

    warped_image = np.ones_like(image)
    warped_image[vx, vy] = image[tuple(transform_map.astype(np.int16))]

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
