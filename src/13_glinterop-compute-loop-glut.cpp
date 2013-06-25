//OpenGL-CL interop: full example with textures and compute loop
//Author: Ugo Varetto

//Requires GLUT and GLM, to deal with the missing support for matrix stack
//in OpenGL >= 3.3

// g++ ../src/13_glinterop-compute-loop-glut.cpp \
// ../src/gl-cl.cpp \
// -DGL_GLEXT_PROTOTYPES -lglut -lGLEW \
// -I/usr/local/cuda/include -lOpenCL \
// -I/usr/local/glm/include
// -I/usr/local/glew/include/GL
// -L/usr/local/glew/lib

#define __CL_ENABLE_EXCEPTIONS

#include <glew.h>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath> //isinf


#include <GL/freeglut.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//OpenCL C++ wrapper
#include "cl.hpp"

#define gle std::cout << "[GL] - " \
                      << __LINE__ << ' ' << glGetError() << std::endl;

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
 
    if(logsize > 1) {
        std::cout << "Vertex shader:\n";
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(vs, logsize, 0, &errmsg[0]);
        std::cout << &errmsg[0] << std::endl;
    }
    // Compile Fragment Shader
    glShaderSource(fs, 1, &fragmentSrc, 0);
    glCompileShader(fs);

    // Check Fragment Shader
    glGetShaderiv(fs, GL_COMPILE_STATUS, &res);
    glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &logsize);
    if(logsize > 1) {
        std::cout << "Fragment shader:\n";
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(fs, logsize, 0, &errmsg[0]);
        std::cout << &errmsg[0] << std::endl;
    }

    // Link the program
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    // Check the program
    glGetProgramiv(program, GL_LINK_STATUS, &res);
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logsize);
    if(logsize > 1) {
        std::cout << "GLSL program:\n";
        std::vector<char> errmsg(logsize + 1, 0);
        glGetShaderInfoLog(program, logsize, 0, &errmsg[0]);
        std::cout << &errmsg[0] << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}


//------------------------------------------------------------------------------
std::vector< float > create_2d_grid(int width, int height,
                                     int xOffset, int yOffset,
                                     float value) {
    std::vector< float > g(width * height);
    for(int y = 0; y != height; ++y) {
        for(int x = 0; x != width; ++x) {
            if(y < yOffset
               || x < xOffset
               || y >= height - yOffset
               || x >= width - xOffset) g[y * width + x] = value;
            else g[y * width + x] = float(0);
        }
    }
    return g;
}

//NOTE: it is important to keep the \n eol at the end of each line
//      to be able to easily match the line reported in the comiler
//      error to the location in the source code

//------------------------------------------------------------------------------
const char kernelSrc[] =
    "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |\n"
    "                               CLK_FILTER_NEAREST |\n"
    "                               CLK_ADDRESS_NONE;\n"
    "float laplacian(read_only image2d_t src, int2 c) {\n"
    "   const float v = (read_imagef(src, sampler, c)).x;\n"
    "   const float n = (read_imagef(src, sampler, c + (int2)( 0,-1))).x;\n"
    "   const float s = (read_imagef(src, sampler, c + (int2)( 0, 1))).x;\n"
    "   const float e = (read_imagef(src, sampler, c + (int2)( 1, 0))).x;\n"
    "   const float w = (read_imagef(src, sampler, c + (int2)(-1, 0))).x;\n"
    "   return (n + s + e + w - 4.0f * v);\n"
    "}\n"
    // "__kernel void apply_stencil(read_only image2d_t src,\n"
    // "                            write_only image2d_t out,\n"
    // "                            float DIFFUSION_SPEED) {\n"
    // "   const int2 center = (int2)(get_global_id(0) + 2, \n"
    // "                              get_global_id(1) + 2);\n"
    // "   const float v = (read_imagef(src, sampler, center)).x;\n"
    // "   const float n = laplacian(src, center + (int2)(0, 1));\n"
    // "   const float s = laplacian(src, center + (int2)(0, -1));\n"
    // "   const float e = laplacian(src, center + (int2)(-1, 0));\n"
    // "   const float w = laplacian(src, center + (int2)(1, 0));\n"
    // "   const float f = v + DIFFUSION_SPEED * (n + s + e + w - 4.0f * v);\n"
    // "   write_imagef(out, center, (float4)(f, 0, 0, 1));\n"
    // "}";
    "__kernel void apply_stencil(read_only image2d_t src,\n"
    "                            write_only image2d_t out,\n"
    "                            float DIFFUSION_SPEED) {\n"
    "   const int2 center = (int2)(get_global_id(0) + 1, \n"
    "                              get_global_id(1) + 1);\n"
    "   const float v = (read_imagef(src, sampler, center)).x;\n"
    "   const float f = v + DIFFUSION_SPEED * laplacian(src, center);\n"
    "   write_imagef(out, center, (float4)(f, 0, 0, 1));\n"
    "}";
const char fragmentShaderSrc[] =  //normalize value to map it to shades of gray
    "#version 330 core\n"
    "in vec2 UV;\n"
    "out vec3 color;\n"
    "uniform sampler2D cltexture;\n"
    "uniform float maxvalue;\n"
    "uniform float minvalue;\n"
    "void main() {\n"
    "  float c = texture2D(cltexture, UV).r;\n"
    "  color = vec3(smoothstep(minvalue, maxvalue, c));\n"
    "}";
const char vertexShaderSrc[] =
    "#version 330 core\n"
    "layout(location = 0) in vec2 pos;\n"
    "layout(location = 1) in vec2 tex;\n"
    "out vec2 UV;\n"
    "uniform mat4 MVP;\n"
    "void main() {\n"
    "  gl_Position = vec4(pos, 0.0f, 1.0f);\n"
    "  UV = tex;\n"
    "}";   


//------------------------------------------------------------------------------
void key_callback(unsigned char key, int xx, int yy) {
    switch(key) {
        case 27: // QUIT
            exit(EXIT_SUCCESS);
        default:
            ;    
    }
}    

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
    if(argc > 1 && argc < 5) {
      std::cout << "usage: " << argv[0]
                << "\n <platform id(0, 1...)>"
                << " <size>\n"
                << " <workgroup size>\n"
                << "   (size - 2) DIV (workgroup size) = 0 (no remainder)\n"
                << " <diffusion speed>\n"
                << " [boundary value; default = 1]"
                << " negative values are rendered with shades of green"
                << std::endl; 
      exit(EXIT_FAILURE);          
    }
    try {
        const int platformID = argc > 1 ? atoi(argv[1]) : 0;
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices;
        cl::Platform::get(&platforms);
        if(platforms.size() <= platformID) {
            std::cerr << "Platform id " << platformID << " is not available\n";
            exit(EXIT_FAILURE);
        }
        platforms[platformID].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
        const int STENCIL_SIZE = 3;
        const int SIZE = argc > 1 ? atoi(argv[2]) : 34;
        const int GLOBAL_WORK_SIZE = SIZE - 2 * (STENCIL_SIZE / 2);
        const int LOCAL_WORK_SIZE = argc > 1 ? atoi(argv[3]) : 4;
        const float DIFFUSION_SPEED = argc > 1 ? atof(argv[4]) : 0.22;
        const float BOUNDARY_VALUE = argc > 5 ? atof(argv[5]) : 1.0f;
//GRAPHICS SETUP     
        glewInit();
        glutInit(&argc, argv);
        glutInitContextVersion(3,3);
        glutInitContextProfile(GLUT_CORE_PROFILE);
        //glutInitContextFlags (GLUT_FORWARD_COMPATIBLE | GLUT_DEBUG);
        
        glutInitWindowSize(640,480);
        glutKeyboardFunc(key_callback);
        glutCreateWindow("OpenCL - GL interop");     
        glutSwapBuffers();

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
        try {
            program.build(devices);
        } catch(const cl::Error& err) {
            std::string s;
            program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &s);           
            std::cout << s << std::endl;
            throw(err);
        }
        cl::Kernel kernel(program, "apply_stencil");

//GEOMETRY AND OPENCL-OPENGL MAPPING
 
        //geometry: textured quad; the texture color value is computed by
        //OpenCL
        float quad[] = {-1.0f,  1.0f,
                         -1.0f, -1.0f,
                          1.0f, -1.0f,
                          1.0f, -1.0f,
                          1.0f,  1.0f,
                         -1.0f,  1.0f};

        float texcoord[] = {0.0f, 1.0f,
                             0.0f, 0.0f,
                             1.0f, 0.0f,
                             1.0f, 0.0f,
                             1.0f, 1.0f,
                             0.0f, 1.0f};                 
        GLuint quadvbo;  
        glGenBuffers(1, &quadvbo);
        glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float),
                     &quad[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        GLuint texbo;  
        glGenBuffers(1, &texbo);
        glBindBuffer(GL_ARRAY_BUFFER, texbo);
        glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float),
                     &texcoord[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0); 


        //create textures mapped to CL buffers; initialize data in textures
        //directly

        std::vector< float > grid = create_2d_grid(SIZE, SIZE,
                                                   STENCIL_SIZE / 2,
                                                   STENCIL_SIZE / 2,
                                                   BOUNDARY_VALUE);
        GLuint texEven;  
        glGenTextures(1, &texEven);

        glBindTexture(GL_TEXTURE_2D, texEven);
        
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_R32F, //IMPORTANT: required for unnormalized values;
                              //without this all the values in the texture
                              //are clamped to [0, 1];
                     SIZE,
                     SIZE,
                     0,
                     GL_RED,
                     GL_FLOAT,
                     &grid[0]);
        //optional
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        //required - use GL_NEAREST instead of GL_LINEAR to visualize
        //the actual discrete pixels
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
       
        glBindTexture(GL_TEXTURE_2D, 0);


        GLuint texOdd;  
        glGenTextures(1, &texOdd);
        glBindTexture(GL_TEXTURE_2D, texOdd);
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_R32F, //IMPORTANT: required for unnormalized values;
                              //without this all the values in the texture
                              //are clamped to [0, 1];
                     SIZE,
                     SIZE,
                     0,
                     GL_RED,
                     GL_FLOAT,
                     &grid[0]);
        //optional
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        //required
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
        glBindTexture(GL_TEXTURE_2D, 0);


        //create CL buffers mapped to textures
        std::vector< cl::Image2DGL > clbuffers(2);
        clbuffers[0] = cl::Image2DGL(context,
                                     CL_MEM_READ_WRITE, 
                                     GL_TEXTURE_2D,
                                     0,
                                     texEven);
        clbuffers[1] = cl::Image2DGL(context,
                                     CL_MEM_READ_WRITE, 
                                     GL_TEXTURE_2D,
                                     0,
                                     texOdd); 

//OPENGL RENDERING SHADERS
        //create opengl rendering program

        GLuint glprogram = create_program(vertexShaderSrc, fragmentShaderSrc);
            
        //extract ids of shader variables
        GLuint mvpID = glGetUniformLocation(glprogram, "MVP");
        GLuint textureID = glGetUniformLocation(glprogram, "cltexture");
        GLuint maxValueID = glGetUniformLocation(glprogram, "maxvalue");
        GLuint minValueID = glGetUniformLocation(glprogram, "minvalue");

        //enable gl program
        glUseProgram(glprogram);

        //set texture id
        glUniform1i(textureID, 0); //always use texture 0

        //set min and max value; required to map it to shades of gray
        glUniform1f(maxValueID, BOUNDARY_VALUE);
        glUniform1f(minValueID, 0.0f);

//COMPUTE AND RENDER LOOP    
        int step = 0;
        GLuint tex = texEven;
        bool converged = false;
        std::cout << std::endl;
        double start = 0;//glfwGetTime();
        double totalTime = 0;
        //make a copy of cl images into a cl::Memory array:
        //the following is required to invoke acquire/release GL object
        //methods in cl::Queue since the methods require pointers to
        //vectors and not (templated) iterators; it is therefore
        //impossible to pass a cl::Image2D vector to the methods
        std::vector<cl::Memory> clmembuffers(clbuffers.begin(),
                                             clbuffers.end());
        float prevError = 0;
        while (!converged) {     

//COMPUTE AND CHECK CONVERGENCE           
            glFinish(); //<-- ensure Open*G*L is done

            //problem(bad design): the enqueue* methods take pointers to 
            //vector<cl::Memory> not templated iterators; it is therefore
            //impossible to reuse the clbuffers vector since a sequence
            //containing a derived type has no relationship with a 
            //sequence of base types
            queue.enqueueAcquireGLObjects(&clmembuffers);
            
            if(IS_EVEN(step)) {
                kernel.setArg(0, clbuffers[0]);
                kernel.setArg(1, clbuffers[1]);
                tex = texOdd;
            } else {//even
                kernel.setArg(0, clbuffers[1]);
                kernel.setArg(1, clbuffers[0]);
                tex = texEven;
            }
            
            kernel.setArg(2, DIFFUSION_SPEED);

            queue.enqueueNDRangeKernel(kernel,
                                       cl::NDRange(0, 0),
                                       cl::NDRange(GLOBAL_WORK_SIZE,
                                                   GLOBAL_WORK_SIZE),
                                       cl::NDRange(LOCAL_WORK_SIZE,
                                                   LOCAL_WORK_SIZE));
            //CHECK FOR CONVERGENCE: extract element at grid center
            //and exit if |element value - boundary value| <= EPS    
            float centerOut = -BOUNDARY_VALUE;
            int activeBuffer = IS_EVEN(step) ? 1 : 0;
            cl::size_t<3> origin;
            origin[0] = SIZE / 2;
            origin[1] = SIZE / 2;
            origin[2] = 0;
            cl::size_t<3> region;
            region[0] = 1;
            region[1] = 1;
            region[2] = 1;
            queue.enqueueReadImage(clbuffers[activeBuffer],
                                   CL_TRUE,
                                   origin,
                                   region,
                                   0, //row pitch; zero for delegating
                                      //computation to OpenCL
                                   0, //slice pitch: for 3D only
                                   &centerOut);
            const double elapsed = 1;//glfwGetTime() - start;
            totalTime += elapsed;
            start = elapsed;
            const float MAX_RELATIVE_ERROR = 0.01;//1%
            const float relative_error =
                fabs(centerOut - BOUNDARY_VALUE) / BOUNDARY_VALUE;
                
            const double error_rate = -(relative_error - prevError) / elapsed;
            if(relative_error <= MAX_RELATIVE_ERROR) converged = true;
            prevError = relative_error; 
            queue.enqueueReleaseGLObjects(&clmembuffers);         
            queue.finish(); //<-- ensure Open*C*L is done          
//RENDER
            // Clear the screen
            glClear(GL_COLOR_BUFFER_BIT);
        
            const int width = glutGet(GLUT_WINDOW_WIDTH);
            const int height = glutGet(GLUT_WINDOW_HEIGHT);
            glViewport(0, 0, width, height);
            //setup OpenGL matrices: no more matrix stack in OpenGL >= 3 core
            //profile, need to compute modelview and projection matrix manually
            const float ratio = width / float(height);
            const glm::mat4 orthoProj = glm::ortho(-ratio, ratio,
                                                   -1.0f,  1.0f,
                                                    1.0f,  -1.0f);
            const glm::mat4 modelView = glm::mat4(1.0f);
            const glm::mat4 MVP       = orthoProj * modelView;
            glUniformMatrix4fv(mvpID, 1, GL_FALSE, glm::value_ptr(MVP));

            //standard OpenGL core profile rendering
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tex);
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, quadvbo);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(1);
            glBindBuffer(GL_ARRAY_BUFFER, texbo);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glDrawArrays(GL_TRIANGLES, 0, 6);    
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glBindTexture(GL_TEXTURE_2D, 0);
            glutSwapBuffers();
           
            ++step; //next step 

            //if any value is NaN or inf do exit
            if(relative_error != relative_error || error_rate != error_rate) {
                std::cout << "\nNaN" << std::endl;
                exit(EXIT_SUCCESS); //EXIT_FAILURE is for execution errors not
                                    //for errors related to data
            }
            if(isinf(relative_error) || isinf(error_rate)) {
                std::cout << "\ninf" << std::endl;
                exit(EXIT_SUCCESS); //EXIT_FAILURE is for execution errors not
                                    //for errors related to data
            }
            std::cout << "\rstep: " << step 
                      << "   error: " << (100 * relative_error)
                      << " %   speed: " << (100 * error_rate) << " %/s   ";
            std::cout.flush();
            glutMainLoopEvent();
        }

        if(converged) 
            std::cout << "\nConverged in " 
                      << step << " steps"
                      << "  time: " << totalTime / 1E3 << " s"
                      << std::endl;
//CLEANUP
        glDeleteBuffers(1, &quadvbo);
        glDeleteBuffers(1, &texbo);
        glDeleteTextures(1, &texEven);
        glDeleteTextures(1, &texOdd);
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
