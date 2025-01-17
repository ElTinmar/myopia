import numpy as np
from vispy import app, gloo

# Window and canvas setup
canvas = app.Canvas(keys='interactive', size=(2560, 1440), fullscreen=True, position=(1440,0))

# Vertex shader: Just pass through the coordinates, no need for randomization here
vertex_shader = """
attribute vec2 a_position;
varying vec2 v_uv;

void main() {
    v_uv = a_position;  // Pass vertex position as UV coordinates
    gl_Position = vec4(a_position, 0.0, 1.0);  // Convert to NDC
}
"""

# Fragment shader: Generate random positions for disks, update based on time, and wrap around
fragment_shader = """
uniform float u_time;
uniform float u_low_radius;
uniform float u_high_radius;
uniform float u_speed_low;
uniform float u_speed_high;
uniform vec2 u_resolution;
uniform vec2 u_low_positions[25];  // Random positions for N large disks
uniform vec2 u_high_positions[144];  // Random positions for M small disks
uniform float u_low_positions_phase[25];
uniform float u_high_positions_phase[144];
uniform int u_num_low;  // Number of large disks
uniform int u_num_high;  // Number of small disks

varying vec2 v_uv;

float disk(float x, float y, float radius) {
    float dist = sqrt(x * x + y * y);
    return step(dist, radius);
}

void main() {
    float combined = 0.0;

    // Loop over the large disks
    for (int i = 0; i < u_num_low; i++) {
        // Get the position of the current low-frequency disk and update with modulo
        vec2 pos = u_low_positions[i];
        pos.x += u_speed_low * u_time;  // Move to the right over time
        pos.x = mod(pos.x, u_resolution.x);  // Wrap around horizontally
        pos.y = mod(pos.y, u_resolution.y);  // Ensure y stays within bounds

        // Convert from normalized space to screen space
        vec2 uv = v_uv * 0.5 + 0.5;  // Normalize UV to range [0, 1]
        uv *= u_resolution;  // Map to screen space (0, 0) to (width, height)

        // Add the disk to the pattern if inside the radius
        //combined += mod(u_time+u_low_positions_phase[i], 1.0) * disk(uv.x - pos.x, uv.y - pos.y, u_low_radius);
        combined += disk(uv.x - pos.x, uv.y - pos.y, u_low_radius);
    }

    // Loop over the small disks
    for (int i = 0; i < u_num_high; i++) {
        // Get the position of the current high-frequency disk and update with modulo
        vec2 pos = u_high_positions[i];
        pos.x += u_speed_high * u_time;  // Move to the left over time
        pos.x = mod(pos.x, u_resolution.x);  // Wrap around horizontally
        pos.y = mod(pos.y, u_resolution.y);  // Ensure y stays within bounds

        // Convert from normalized space to screen space
        vec2 uv = v_uv * 0.5 + 0.5;  // Normalize UV to range [0, 1]
        uv *= u_resolution;  // Map to screen space (0, 0) to (width, height)

        // Add the disk to the pattern if inside the radius
        //combined += mod(u_time+u_high_positions_phase[i], 1.0) * disk(uv.x - pos.x, uv.y - pos.y, u_high_radius);
        combined += disk(uv.x - pos.x, uv.y - pos.y, u_high_radius);
    }

    // Output final color (grayscale disk pattern)
    gl_FragColor = vec4(vec3(combined), 1.0);
}
"""

# Initialize the OpenGL program
program = gloo.Program(vertex_shader, fragment_shader)

# Set up vertex data for a full-screen quad (to cover the whole canvas)
vertices = np.array([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
], dtype=np.float32)

# Bind vertex data
program['a_position'] = gloo.VertexBuffer(vertices)

# Randomly generate N large and M small disk positions
N = 25  # Number of large disks
M = 144  # Number of small disks
np.random.seed(1)  # For reproducibility

def generate_stratified_positions(n, width, height, randomness_factor=0.1):
    # Calculate the number of rows and columns to divide the area into
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    
    positions = []
    
    for i in range(n):
        # Compute grid cell position
        row = i // cols
        col = i % cols
        
        # Calculate the base position for this grid cell
        x_base = (col + 0.5) / cols
        y_base = (row + 0.5) / rows
        
        # Add random displacement
        x_disp = (np.random.random() - 0.5) * randomness_factor
        y_disp = (np.random.random() - 0.5) * randomness_factor
        
        # Final position (with random displacement)
        x = x_base + x_disp
        y = y_base + y_disp
        
        # Scale to the screen size (width and height)
        x_pos = x * width
        y_pos = y * height
        positions.append([x_pos, y_pos])
    
    return np.array(positions)

# Generate stratified positions for large and small disks
low_positions = generate_stratified_positions(N, 2560, 1440, 0.1)
high_positions = generate_stratified_positions(M, 2560, 1440, 0.9)

# Pass positions to the shader
program['u_low_positions'] = low_positions.astype(np.float32)
program['u_low_positions_phase'] = np.random.rand(N)[:,np.newaxis]
program['u_high_positions'] = high_positions.astype(np.float32)
program['u_high_positions_phase'] = np.random.rand(M)[:,np.newaxis]
program['u_num_low'] = N
program['u_num_high'] = M

# Set uniform values for the shader
program['u_low_radius'] = 150.0  # Radius of low-frequency disks
program['u_high_radius'] = 20.0  # Radius of high-frequency disks
program['u_speed_low'] = 30.0  # Speed of low-frequency motion (slow to the right)
program['u_speed_high'] = -30.0  # Speed of high-frequency motion (fast to the left)
program['u_resolution'] = (2560, 1440)  # Canvas resolution (to help position dots)
program['u_time'] = 0.0  # Initialize time

# Animation callback
def update(event):
    global t
    t += 0.016  # Increment time (60 FPS approximation)
    program['u_time'] = t
    canvas.update()

# Draw callback
@canvas.connect
def on_draw(event):
    gloo.clear('black')  # Clear the canvas
    program.draw('triangle_strip')  # Draw the full-screen quad

# Time initialization and start animation
t = 0.0
timer = app.Timer(interval=1/60, connect=update, start=True)  # 60 FPS timer
canvas.show()

# Run the application
app.run()
