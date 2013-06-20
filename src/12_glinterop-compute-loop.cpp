//OpenGL-CL interop: full example with textures and compute loop
//Author: Ugo Varetto

//Requires GLFW and GLM, to deal with the missing support for matrix stack
//in OpenGL >= 3.3

/////////// IN PROGRESS //////////////////


//g++ ../src/... ../src/gl-cl.cpp -I/usr/local/glfw/include \
// -DGL_GLEXT_PROTOTYPES -L/usr/local/glfw/lib -lglfw \
// -I/usr/local/cuda/include -lOpenCL

#define __CL_ENABLE_EXCEPTIONS

#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//OpenCL C++ wrapper
#include "cl.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdexcept>

#ifdef USE_DOUBLE
typedef double real_t;
const GLenum GL_REAL_T = GL_DOUBLE;
#else
typedef float real_t;
const GLenum GL_REAL_T = GL_FLOAT;
#endif

//------------------------------------------------------------------------------
GLuint create_program(const char* vertexSrc,
                      const char* fragmentSrc) {
    // Create the shaders
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    GLint res = GL_FALSE;
    int logsize = 0;
    // Compile Vertex Shader
    glShaderSource(vs, 1, &vertexSrc , NULL);
    glCompileShader(vs);

    // Check Vertex Shader
    glGetShaderiv(vs, GL_COMPILE_STATUS, &res);
    glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logsize);
    if(logsize > 0){
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(vs, logsize, 0, &errmsg[0]);
        throw std::runtime_error(&errmsg[0]);
    }
    // Compile Fragment Shader
    glShaderSource(fs, 1, &fragmentSrc, 0);
    glCompileShader(fs);

    // Check Fragment Shader
    glGetShaderiv(fs, GL_COMPILE_STATUS, &res);
    glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &logsize);
    if(logsize > 0){
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(fs, logsize, 0, &errmsg[0]);
        throw std::runtime_error(&errmsg[0]);
    }

    // Link the program
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    // Check the program
    glGetProgramiv(program, GL_LINK_STATUS, &res);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logsize);
    if(logsize > 0) {
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(program, logsize, 0, &errmsg[0]);
        throw std::runtime_error(&errmsg[0]);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}


//------------------------------------------------------------------------------
std::vector< real_t > create_2d_grid(int width, int height,
                                     int xOffset, int yOffset) {
    std::vector< real_t > g(width * height);
    srand(time(0));
    for(int y = 0; y != height; ++y) {
        for(int x = 0; x != width; ++x) {
            if(y < yOffset
               || x < xOffset
               || y >= height - yOffset
               || x >= width - xOffset) g[y * width + x] = real_t(1);
            else g[y * width + x] = real_t(0);
        }
    }
    return g;
}

//------------------------------------------------------------------------------
void error_callback(int error, const char* description) {
    std::cerr << description << std::endl;
}

//------------------------------------------------------------------------------
void key_callback(GLFWwindow* window, int key,
                         int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

//------------------------------------------------------------------------------
const char kernelSrc[] =
    "float grid(float* g, int row, int col, int columns) {\n"
    "  return g[row *columns + col];\n"
    "}\n"
    "__kernel void apply_stencil(const __global float* in,\n"
    "                            __global float* out,\n"
    "                            int size) {\n"
    "   const int c = get_global_id(0);\n"
    "   const int r = get_global_id(1);\n"
    "   const float v = grid(in, r, c);\n"
    "   const float n = grid(in, r - 1, c);\n"
    "   const float s = grid(in, r + 1, c);\n"
    "   const float e = grid(in, r, c - 1);\n"
    "   const float w = grid(in, r, c + 1);\n"
    "   out[size * r + c] = v + 0.1f*(-4.f * (n + s + e + w));\n"
    "}";
const char fragmentShaderSrc[] =
    "in vec2 UV;\n"
    "out vec3 color;\n"
    "uniform sampler2D cltexture;\n"
    "void main() {\n"
    "  float v = texture2D(cltexture, UV).r;\n"
    "  color = vec3(v, v, v);\n"
    "}"
const char vertexShaderSrc[] =
    "layout(location = 0) in vec2 position;\n"
    "layout(location = 1) in vec2 texcoord;\n"
    "out vec2 UV;\n"
    "uniform mat4 MVP;\n"
    "void main() {\n"
    "  gl_Position = MVP * vec4(position, 0, 1);\n"
    "  UV = texcoord;\n"
    "}"   

//------------------------------------------------------------------------------
bool IS_EVEN(int v) { return v % 2 == 0; }

//------------------------------------------------------------------------------
typedef std::vector< cl_context_properties > CLContextProperties;

//declare external function
CLContextProperties
create_cl_gl_interop_properties(cl_platform_id platform); 

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
//USER INPUT
    if(argc < 4) {
      std::cout << "usage: " << argv[0]
                << " <platform id(0, 1...)>"
                << " <size>"
                << " <workgroup size>"
                << std::endl; 
      exit(EXIT_FAILURE);          
    }
    const bool DONT_NORMALIZE = (argc > 2);
    try {
        const int platformID = atoi(argv[1]);
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        cl::Platform::get(&platforms);
        if(platforms.size() <= platformID) {
            std::cerr << "Platform id " << platformID << " is not available\n";
            exit(EXIT_FAILURE);
        }
        platforms[platformID].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
        const int STENCIL_SIZE = 3;
        const int SIZE = atoi(argv[2]);
        const int GLOBAL_WORK_SIZE = SIZE - 2 * (STENCIL_SIZE / 2);
        const int LOCAL_WORK_SIZE = atoi(argv[3]);

//GRAPHICS SETUP        
        glfwSetErrorCallback(error_callback);

        if(!glfwInit()) {
            std::cerr << "ERROR - glfwInit" << std::endl;
            exit(EXIT_FAILURE);
        }

        glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
        glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
        glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(640, 480,
                                              "Simple example", NULL, NULL);
        if (!window) {
            std::cerr << "ERROR - glfwCreateWindow" << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        glfwSetKeyCallback(window, key_callback);

//OPENCL SETUP        
        //OpenCL context
        CLContextProperties prop = 
            create_cl_gl_interop_properties(platforms[platformID]());
        cl::Context context(devices, &prop[0]);

        cl::CommandQueue queue(context, devices[0]);
       
        //cl kernel
        cl::Program::Sources source(1,
                                    std::make_pair(kernelSrc,
                                                   sizeof(kernelSrc)));
        cl::Program program(context, source);
        program.build(devices);          
       
        cl::Kernel kernel(program, "apply_stencil");

//GEOMETRY AND OPENCL-OPENGL MAPPING
 
        //geometry: textured quad; the texture color is conputed by
        //OpenCL
        real_t quad[] = {-1.0f,  1.0f,
                         -1.0f, -1.0f,
                          1.0f, -1.0f,
                          1.0f, -1.0f,
                          1.0f,  1.0f
                         -1.0f,  1.0f};

        real_t texcoord[] = {0.0f, 1.0f,
                             0.0f, 0.0f,
                             1,0f, 0.0f,
                             1.0f, 0.0f,
                             1.0f, 1.0f,
                             0.0f, 1.0f};                 
        GLuint quadvbo;  
        glGenBuffers(1, &quadvbo);
        glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(real_t),
                     &quad[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        GLuint texbo;  
        glGenBuffers(1, &texbo);
        glBindBuffer(GL_ARRAY_BUFFER, texbo);
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(real_t),
                     &texbo[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 

        //create textures mapped to CL buffers
        GLuint texEven;  
        glGenTextures(1, &texEven);
        glBindTexture(GL_TEXTURE_2D, textEven);
        glTexImage2D(GL_TEXTURE_2D,
                      0,
                      GL_LUMINANCE,
                      SIZE,
                      SIZE,
                      0,
                      GL_LUMINANCE,
                      GL_FLOAT,
                      0);
        glBindTexture(0);

        GLuint texOdd;  
        glGenTextures(1, &texEven);
        glBindTexture(GL_TEXTURE_2D, texOdd);
        glTexImage2D(GL_TEXTURE_2D,
                      0,
                      GL_LUMINANCE,
                      SIZE,
                      SIZE,
                      0,
                      GL_LUMINANCE,
                      GL_FLOAT,
                      0);
        glBindTexture(0);


        //create CL buffers mapped to textures
        cl_int status;
        cl_mem clbufferEven = clCreateFromGLTexture2D(context(),
                                                      CL_MEM_READ_WRITE, 
                                                      GL_TEXTURE_2D,
                                                      0,
                                                      texEven,
                                                      &status);

        if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clCreateFromGLTexture2D");
        cl_mem clbufferOdd = clCreateFromGLTexture2D(context(),
                                                      CL_MEM_READ_WRITE, 
                                                      GL_TEXTURE_2D,
                                                      0,
                                                      texOdd,
                                                      &status);
        if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clCreateFromGLTexture2D");    

        //init data
        std::vector< real_t > data = create_2d_grid(SIZE, SIZE,
                                                    STENCIL_SIZE / 2,
                                                    STENCIL_SIZE / 2 );
        queue.enqueueWriteBuffer(clbufferEven,
                                 CL_TRUE,
                                 0,
                                 data.size() * sizeof(real_t),
                                 &data[0]);

        queue.enqueueWriteBuffer(clbufferOdd,
                                 CL_TRUE,
                                 0,
                                 data.size() * sizeof(real_t),
                                 &data[0]);
        
//OPENGL RENDERING SHADERS
        //create opengl rendering program
        GLuint glprogram = create_program(vertexShaderSrc, fragmentShaderSrc);
            
        //extract ids of shader variables
        GLuint mvpID = glGetUniformLocation(glprogram, "MVP");
        GLuint textureID = glGetUniformLocation(glprogram, "cltexture");

        //enable gl program
        glUseProgram(glprogram);

        //set texture id
        glUniform1i(textureID, 0); //always use texture 0

//COMPUTE AND RENDER LOOP    
        int step = 0;
        GLuint tex = texEven;
        //rendering & simulation loop
        while (!glfwWindowShouldClose(window)) {     

//COMPUTE 
            glFinish(); //<-- ensure Open*G*L is done
            //acquire CL objects and perform computation step
            status = clEnqueueAcquireGLObjects(queue(),
                                                      1,
                                                      &clbufferEven, 0, 0, 0);
            if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clEnqueueAcquireGLObjects");
            cl_int status = clEnqueueAcquireGLObjects(queue(),
                                                      1,
                                                      &clbufferOdd, 0, 0, 0);
            if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clEnqueueAcquireGLObjects");      
            
            if(IS_EVEN(step)) {
                status = clSetKernelArg(kernel(), //kernel
                                        0,      //parameter id
                                        sizeof(cl_mem), //size of parameter
                                        &clbufferEven); //pointer to parameter
            
                if(status != CL_SUCCESS )
                    throw std::runtime_error("ERROR - clSetKernelArg");
                status = clSetKernelArg(kernel(), //kernel
                                        1,      //parameter id
                                        sizeof(cl_mem), //size of parameter
                                        &clbufferOdd); //pointer to parameter
            
                if(status != CL_SUCCESS )
                    throw std::runtime_error("ERROR - clSetKernelArg");
                tex = texEven;
            } else {//even
                status = clSetKernelArg(kernel(), //kernel
                                        0,      //parameter id
                                        sizeof(cl_mem), //size of parameter
                                        &clbufferOdd); //pointer to parameter
            
                if(status != CL_SUCCESS )
                    throw std::runtime_error("ERROR - clSetKernelArg");
                status = clSetKernelArg(kernel(), //kernel
                                        1,      //parameter id
                                        sizeof(cl_mem), //size of parameter
                                        &clbufferEven); //pointer to parameter
            
                if(status != CL_SUCCESS )
                    throw std::runtime_error("ERROR - clSetKernelArg");
                tex = texOdd;                 
            }
            
            queue.setKernelArg(2, SIZE);
            
            queue.enqueueNDRangeKernel(kernel,
                                       cl::NDRange(0, 0),
                                       cl::NDRange(GLOBAL_WORK_SIZE,
                                                   GLOBAL_WORK_SIZE),
                                       cl::NDRange(LOCAL_WORK_SIZE,
                                                   LOCAL_WORK_SIZE));
            
            status = clEnqueueReleaseGLObjects(queue(),
                                               1, &clbufferEven, 0, 0, 0);
            if(status != CL_SUCCESS)
                throw std::runtime_error("ERROR - clEnqueueReleaseGLObjects");
            status = clEnqueueReleaseGLObjects(queue(),
                                               1, &clbufferOdd, 0, 0, 0);
            if(status != CL_SUCCESS)
                throw std::runtime_error("ERROR - clEnqueueReleaseGLObjects");         
            
            queue.finish(); //<-- ensure Open*C*L is done
//RENDER
            //setup OpenGL matrices: no more matrix stack in OpenGL >= 3 core
            //profile, need to compute modelview and projection matrix manually
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            const float ratio = width / float(height);
            glm::mat4 orthoProj = glm::perspective(-ratio, ratio,
                                                   -1.0f,  1.0f,
                                                    1.0f, -1.0f);
            glm::mat4 modelView = glm::mat4(1.0f);
            glm::mat4 MVP       = orthoProj * modelView;
            glUniformMatrix4fv(mvpID, 1, GL_FALSE, &MVP[0][0]);

            //standard OpenGL core profile rendering
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex);
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
            glVertexAttribPointer(0, 2, GL_REAL_T, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, texcoords);
            glVertexAttribPointer(1, 2, GL_REAL_T, GL_FALSE, 0, 0);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindTexture(0); 
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glBindTexture(GL_TEXTURE_2D, 0);  
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

//CLEANUP
        glDeleteBuffers(1, &quadvbo);
        glDeleteBuffers(1, &texbo);
        glDeleteTextures(1, &texEven);
        glDeleteTextures(1, &texOdd);
        glfwDestroyWindow(window);

        clReleaseMemObject(clbufferOdd);
        clReleaseMemObject(clbufferEven);

        glfwTerminate();
        exit(EXIT_SUCCESS);
    } catch(const cl::Error& e) {
        std::cerr << e.what() << ": Error code " << e.err() << std::endl;   
        exit(EXIT_FAILURE);
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
