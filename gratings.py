import numpy as np
from vispy import app, gloo

# Window and canvas setup
canvas = app.Canvas(keys='interactive', size=(2560, 1440), fullscreen=True, position=(1440,0))

# Vertex shader: Passes the vertex coordinates directly
vertex_shader = """
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
    v_uv = (a_position + 1.0) / 2.0;  // Normalize coordinates to range [0, 1]
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

# Fragment shader: Generates the dual-frequency pattern
fragment_shader = """
uniform float u_time;
uniform float u_low_freq;
uniform float u_high_freq;
uniform float u_speed_low;
uniform float u_speed_high;

varying vec2 v_uv;

void main() {
    // Calculate horizontal positions
    float x = v_uv.x * 2.0 - 1.0;  // Scale UV coordinates to [-1, 1]

    // Low-frequency component
    float low_pattern = sin(2.0 * 3.14159 * (u_low_freq * x + u_speed_low * u_time));

    // High-frequency component
    float high_pattern = sin(2.0 * 3.14159 * (u_high_freq * x + u_speed_high * u_time));

    // Combine patterns
    float combined = 0.5 * low_pattern + 0.5 * high_pattern;

    // Convert to grayscale and output
    gl_FragColor = vec4(vec3(combined * 0.5 + 0.5), 1.0);  // Map [-1, 1] to [0, 1]
}
"""

# Initialize the OpenGL program
program = gloo.Program(vertex_shader, fragment_shader)

# Set up vertex data for a full-screen quad
vertices = np.array([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
], dtype=np.float32)

# Bind vertex data
program['a_position'] = gloo.VertexBuffer(vertices)

# Set uniform values for the shader
program['u_low_freq'] = 1  # Low spatial frequency
program['u_high_freq'] = 8  # High spatial frequency
program['u_speed_low'] = 0.2  # Speed of low-frequency motion
program['u_speed_high'] = -0.5  # Speed of high-frequency motion
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
