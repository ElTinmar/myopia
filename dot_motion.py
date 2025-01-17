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
uniform vec2 u_low_positions[50];  // Random positions for N large disks
uniform vec2 u_high_positions[100];  // Random positions for M small disks
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
N = 50  # Number of large disks
M = 100  # Number of small disks
np.random.seed(42)  # For reproducibility

# Random positions for N large disks in screen coordinates (range [0, width], [0, height])
low_positions = np.random.rand(N, 2) * [2560, 1440]  # Range [0, 600] for x and y
# Random positions for M small disks in screen coordinates (range [0, width], [0, height])
high_positions = np.random.rand(M, 2) * [2560, 1440]  # Range [0, 600] for x and y

# Pass positions to the shader
program['u_low_positions'] = low_positions.astype(np.float32)
program['u_high_positions'] = high_positions.astype(np.float32)
program['u_num_low'] = N
program['u_num_high'] = M

# Set uniform values for the shader
program['u_low_radius'] = 100.0  # Radius of low-frequency disks
program['u_high_radius'] = 20.0  # Radius of high-frequency disks
program['u_speed_low'] = 20.0  # Speed of low-frequency motion (slow to the right)
program['u_speed_high'] = -20.0  # Speed of high-frequency motion (fast to the left)
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
