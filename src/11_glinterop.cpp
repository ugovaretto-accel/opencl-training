//OpenGL-CL interop
//Author: Ugo Varetto
//g++ ../src/11_glinterop.cpp ../src/gl-cl.cpp -I/usr/local/glfw/include \
// -DGL_GLEXT_PROTOTYPES -L/usr/local/glfw/lib -lglfw \
// -I/usr/local/cuda/include -lOpenCL
// [-DUSE_DOUBLE] 

//Use double precision on OpenGL as well!


#define __CL_ENABLE_EXCEPTIONS

#include <GLFW/glfw3.h>
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
void error_callback(int error, const char* description) {
    std::cerr << description << std::endl;
}

//------------------------------------------------------------------------------
void key_callback(GLFWwindow* window, int key,
                         int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

//#define DONT_NORMALIZE to show the effect of accumulated errors in output
//array: the triangle keeps on shrinking in size
const char kernelSrc[] =
    "#ifdef USE_DOUBLE\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "typedef double real_t;\n"
    "typedef double4 real_t4;\n"
    "#else\n"
    "typedef float  real_t;\n"
    "typedef float4 real_t4;\n"
    "#endif\n" 
    "__kernel void rotate_vertex(__global real_t4* vertices,\n"
    "                            real_t time) {\n"
    "   const int vid = get_global_id(0);\n"    
    "   real_t4 v = vertices[vid];\n"
    "#ifndef DONT_NORMALIZE\n"
    "   real_t m = sqrt(v.x*v.x + v.y*v.y);\n"
    "#endif\n"
    "   real_t c;\n"
    "   real_t s = sincos(time / 300.0f, &c);\n"
    "   v.x = -v.y*s + v.x*c;\n"
    "   v.y = v.x*s + v.y*c;\n"
    "#ifndef DONT_NORMALIZE\n"
    "   v.xy /= sqrt(v.x*v.x + v.y*v.y);\n"
    "   v.xy *= m;\n"
    "#endif\n"
    "   vertices[vid] = v;\n"   
    "}";

typedef std::vector< cl_context_properties > CLContextProperties;

//declare external function
CLContextProperties
create_cl_gl_interop_properties(cl_platform_id platform); 

//------------------------------------------------------------------------------
int main(int argc, char** argv) {

    if(argc < 2) {
      std::cout << "usage: " << argv[0]
                << " <platform id(0, 1...)>"
                << " [nonormalize] - add this to watch the effects of"
                   " accumulated errors"
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
       

        glfwSetErrorCallback(error_callback);

        if (!glfwInit())
            exit(EXIT_FAILURE);

        GLFWwindow* window = glfwCreateWindow(640, 480,
                                              "Simple example", NULL, NULL);
        if (!window) {
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        //create OpenCL context for OpenGL-CL interoperability *after* 
        //OpenGL context 
        //use () operator to have the cl::Platform object return the actual
        //cl_platform_id value
        CLContextProperties prop = 
            create_cl_gl_interop_properties(platforms[platformID] () // <--
                                           );
        cl::Context context(devices, &prop[0]);

        cl::CommandQueue queue(context, devices[0]);

        glfwSetKeyCallback(window, key_callback);
        //cl kernel
        cl::Program::Sources source(1,
                                    std::make_pair(kernelSrc,
                                                   sizeof(kernelSrc)));
        cl::Program program(context, source);
#ifdef USE_DOUBLE
        if(DONT_NORMALIZE) program.build(devices,
                                         "-DDONT_NORMALIZE -DUSE_DOUBLE");
        else program.build(devices, "-DUSE_DOUBLE");    
#else
        if(DONT_NORMALIZE) program.build(devices, "-DDONT_NORMALIZE");
        else program.build(devices);    
#endif        
       
       
        cl::Kernel kernel(program, "rotate_vertex");
 

        //geometry
        real_t vertices[] = {-0.6f, -0.4f, 0.f, 1.0f,
                             0.6f, -0.4f, 0.f, 1.0f,
                             0.f,   0.6f, 0.f, 1.0f};

        GLuint vbo;
        cl_mem clbuffer;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(real_t),
                     &vertices[0], GL_STATIC_DRAW);
        clbuffer = clCreateFromGLBuffer(context(), CL_MEM_READ_WRITE, vbo, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 

        //rendering loop
        while (!glfwWindowShouldClose(window)) {
            real_t ratio;
            int width, height;

            glfwGetFramebufferSize(window, &width, &height);
            ratio = width / real_t(height);

            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
            glMatrixMode(GL_MODELVIEW);


            real_t elapsedTime = real_t(glfwGetTime()); 
            glLoadIdentity();
            //use OpenCL kernel to update vertices instead of following line
            //glRotatef(elapsedTime * 50.f, 0.f, 0.f, 1.f);
            
            //actual processing:
            //0) ensure all pending OpenGL commands are completed
            //1) acquire OpenGL resources to make them available in OpenCL
            //2) run OpenCL kernels
            //3) release OpenGL-->OpenCL mapping
            //4) ensure all pending OpenCL commands are completed

            //NB: no C++ wrappers exist for OpenGL-OpenCL interop; use
            //    standard C functions with proper error checking

            glFinish(); //<-- ensure Open*G*L is done
            cl_int status = clEnqueueAcquireGLObjects(queue(),
                                                      1,
                                                      &clbuffer, 0, 0, 0);
            if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clEnqueueAcquireGLObjects");  
                
            status = clSetKernelArg(kernel(), //kernel
                                    0,      //parameter id
                                    sizeof(cl_mem), //size of parameter
                                    &clbuffer); //pointer to parameter
            
            if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clSetKernelArg");
            
            kernel.setArg(1, elapsedTime);
            queue.enqueueNDRangeKernel(kernel,
                                       cl::NDRange(0),
                                       cl::NDRange(3), //3 4D elements
                                       cl::NDRange(1));
            
            status = clEnqueueReleaseGLObjects(queue(),
                                               1, &clbuffer, 0, 0, 0);
            if(status != CL_SUCCESS)
                throw std::runtime_error("ERROR - clEnqueueReleaseGLObjects");     
            queue.finish(); //<-- ensure Open*C*L is done

            //standard OpenGL core profile rendering(no shaders specified)
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(0, 4, GL_REAL_T, GL_FALSE, 0, 0);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindBuffer(GL_ARRAY_BUFFER, 0); 
            glDisableVertexAttribArray(0);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        glDeleteBuffers(1, &vbo);
        glfwDestroyWindow(window);

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
