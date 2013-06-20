//OpenGL-CL interop: full example with textures and compute loop
//Author: Ugo Varetto

/////////// IN PROGRESS //////////////////


//g++ ../src/... ../src/gl-cl.cpp -I/usr/local/glfw/include \
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

//------------------------------------------------------------------------------
void resize(GLFWindow*, int width, int height) {
    const real_t ratio = width / real_t(height);

    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(float(-ratio), float(ratio), -1.f, 1.f, 1.f, -1.f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
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
            std::cerr << "ERROR - glfwInit" << std::endl;
            exit(EXIT_FAILURE);

        GLFWwindow* window = glfwCreateWindow(640, 480,
                                              "Simple example", NULL, NULL);
        if (!window) {
            std::cerr << "ERROR - glfwCreateWindow" << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwMakeContextCurrent(window);

        glfwSetKeyCallback(window, key_callback);

        glfwSetFrameBufferSizeCallback(window, resize);
        //create OpenCL context for OpenGL-CL interoperability *after* 
        //OpenGL context 
        //use () operator to have the cl::Platform object return the actual
        //cl_platform_id value
        CLContextProperties prop = 
            create_cl_gl_interop_properties(platforms[platformID]());
        cl::Context context(devices, &prop[0]);

        cl::CommandQueue queue(context, devices[0]);

       
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
       
       
        cl::Kernel kernel(program, "apply_stencil");
 
        //geometry: textured quad; the texture color is conputed by
        //OpenCL
        real_t quad[] = {-1.0f,  1.0f,
                         -1.0f, -1.0f,
                          1.0f, -1.0f,
                          1.0f, -1.0f,
                          1.0f,  1.0f
                         -1.0f,  1.0f};

        real_t texcoord = {0.0f, 1.0f,
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




        //create pixel buffer object mapped to output OpenCL buffers for
        //generating values and to 2d textures for reading data from
        //fragment shaders
        GLuint pboEven;  
        glGenBuffers(1, &pboEven);
        glBindBuffer(GL_ARRAY_BUFFER, pboEven);
        glBufferData(GL_ARRAY_BUFFER,
                     GRID_BYTE_SIZE,
                     0,
                     GL_STATIC_DRAW);
                     &quad[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        GLuint pboOdd;  
        glGenBuffers(1, &pboOdd);
        glBindBuffer(GL_ARRAY_BUFFER, pboOdd);
        glBufferData(GL_ARRAY_BUFFER,
                     GRID_BYTE_SIZE,
                     0,
                     GL_STATIC_DRAW);
                     &quad[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);


        
        cl_mem clbufferEven = clCreateFromGLBuffer(context(),
                                                   CL_MEM_READ_WRITE,
                                                   pboEven, &status);
        if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clCreateFromGLBuffer");
        cl_mem clbufferOdd = clCreateFromGLBuffer(context(),
                                                   CL_MEM_READ_WRITE,
                                                   pboOdd, &status);
        if(status != CL_SUCCESS )
                throw std::runtime_error("ERROR - clCreateFromGLBuffer");    

        //init data on GPU

        do {
            //set input and output buffers
            //invoke kernel

        }while (!glfwWindowShouldClose(window));
        

        int step = 0;
        GLuint pbo = pboEven;

        //rendering loop
        while (!glfwWindowShouldClose(window)) {
            real_t start = real_t(glfwGetTime()); 
           
            glFinish(); //<-- ensure Open*G*L is done
            cl_int status = clEnqueueAcquireGLObjects(queue(),
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
                pbo = pboEven;
            } else {//even
                status = clSetKernelArg(kernel(), //kernel
                                        0,      //parameter id
                                        sizeof(cl_mem), //size of parameter
                                        &clbufferOdd); //pointer to parameter
            
                if(status != CL_SUCCESS )
                    throw std::runtime_error("ERROR - clSetKernelArg");
                status = clSetKernelArg(kernel(), //kernel
                                        0,      //parameter id
                                        sizeof(cl_mem), //size of parameter
                                        &clbufferEven); //pointer to parameter
            
                if(status != CL_SUCCESS )
                    throw std::runtime_error("ERROR - clSetKernelArg");
                pbo = pboOdd;                 
            }

            
            queue.enqueueNDRangeKernel(kernel,
                                       cl::NDRange(0, 0),
                                       cl::NDRange(SIZE, SIZE),
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

            //standard OpenGL core profile rendering

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height,
                         0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(0, 2, GL_REAL_T, GL_FALSE, 0, 0);
            glDrawArrays(GL_TRIANGLES, 0, 6);
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
