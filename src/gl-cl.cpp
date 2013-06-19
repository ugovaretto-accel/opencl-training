//OpenCL/GL utility functions
//Author: Ugo Varetto

#include <vector>
#include <stdexcept>

#ifdef __APPLE__
 #include <OpenGL/OpenGL.h>
#else
 #ifdef WIN32
  #include <wingdi.h>
 #else
  #include <GL/glx.h>
 #endif
#endif 

#ifdef __APPLE__
 #include <OpenCL/cl.h>
 #include <OpenCL/cl_gl.h> 
#else
 #include <CL/cl.h>
 #include <CL/cl_gl.h> 
#endif


typedef std::vector< cl_context_properties > CLProperties;

//when using the C++ API pass cl::Platform::operator()() to access
//the wrapped cl_platform_id resource

CLProperties create_cl_gl_interop_properties(cl_platform_id platform) {
#if defined (__APPLE__) || defined(MACOSX)
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
    cl_context_properties props[] = {
        CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
        (cl_context_properties)kCGLShareGroup,
        0
    };
    return CLProperties(props, props 
                        + sizeof(props) / sizeof(cl_context_properties));
#else
    #if defined WIN32 
        cl_context_properties props[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
            CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0
        };
        return CLProperties(props, props 
                            + sizeof(props) / sizeof(cl_context_properties));
    #else
        cl_context_properties props[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0
        };
        return CLProperties(props, props 
                            + sizeof(props) / sizeof(cl_context_properties));
    #endif
#endif
}