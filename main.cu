#include <iostream>
#include <cstdlib>
#include <execution>

#include "event.h"
#include "status.h"

class memory_resource{
  virtual void* do_allocate(size_t Size, size_t Alignment, void* Previous) noexcept = 0;
  virtual void do_deallocate(void* Pointer) noexcept = 0;

  virtual bool do_is_equal(const memory_resource& Other) const noexcept{
    return this == &Other;
  }
public:

};

class allocator{
  void*(*allocate)(size_t Bytes);
  void(*deallocate)(void* Pointer);
};


namespace detail{
  struct address_control_block{

  };
}

template <typename T>
class uniform_address{
  template <typename>
  friend class host_pointer;
  template <typename>
  friend class device_pointer;

  T* DevicePointer;
  T* HostPointer;
public:

};

template <typename T>
class host_pointer{
  T* Pointer;
public:
  T* operator->() const noexcept{
    return Pointer;
  }
  T& operator*() const noexcept{
    return *Pointer;
  }
};
template <typename T>
class device_pointer{
  T* Pointer;
public:
  __device__ T* operator->() const noexcept{
    return Pointer;
  }
  __device__ T& operator*() const noexcept{
    return *Pointer;
  }
};


class error{};
class status_code{
public:
  struct domain{
    cu::string_view(*name)();
    cu::string_view(*message)(int64_t Value);
    bool(*equivalent)(const domain* Other, )
  };

  status_code();

  int64_t value() const noexcept;
  cu::string_view message() const noexcept;

private:
  int64_t Value;
  const domain* Domain;
};




template <typename T>
class owned_ptr{
  struct deleter{

  };
public:
  using pointer = T*;
  using reference = T&;
  using const_pointer = std::conditional_t<std::is_pointer_v<pointer>, pointer, const pointer&>;

  owned_ptr() = default;
  owned_ptr(const_pointer Ptr) noexcept : Ptr(Ptr){}
  owned_ptr(std::nullptr_t) noexcept {}
  owned_ptr(const owned_ptr&) = delete;
  owned_ptr(owned_ptr&& Other) noexcept : Ptr(Other.release()){}

  ~owned_ptr(){

  }

  owned_ptr& operator=(const owned_ptr&) = delete;
  owned_ptr& operator=(owned_ptr&& Other) noexcept{
    if(Other.Ptr != Ptr)
      reset(Other.release());
    return *this;
  }

  reference operator*() const noexcept{
    return *Ptr;
  }
  pointer operator->() const noexcept{
    return Ptr;
  }

  pointer get() const noexcept{
    return Ptr;
  }
  pointer release(){
    pointer Tmp = Ptr;
    Ptr = nullptr;
    return Tmp;
  }
  void reset(pointer Pointer = nullptr){
    (*this)(Ptr);
    Ptr = Pointer;
  }

private:
  pointer Ptr = nullptr;
};
template <typename T>
class borrowed_ptr{
public:
  borrowed_ptr() = default;
  borrowed_ptr(const owned_ptr<T>)
private:
  T* Pointer;
};








class CUioStream{
  char* Buffer;
public:

};




struct memory_range{
  CUdeviceptr Address;
  size_t Size;
};

std::ostream& operator<<(std::ostream& OS, CUmemAllocationHandleType HandleTypes){
  OS << "{ ";
  bool Flag = false;
  if(HandleTypes & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    OS << "posix_fd | ";
    Flag = true;
  }
  if(HandleTypes & CU_MEM_HANDLE_TYPE_WIN32) {
    OS << "win32 | ";
    Flag = true;
  }
  if(HandleTypes & CU_MEM_HANDLE_TYPE_WIN32_KMT)
    OS << "win32_kmt ";
  if(Flag)
    OS << "\b\b";
  return OS << "}";
}
std::ostream& operator<<(std::ostream& OS, CUmemorytype MemoryType){
  switch(MemoryType){
    case CU_MEMORYTYPE_HOST:
      return OS << "host";
    case CU_MEMORYTYPE_DEVICE:
      return OS << "device";
    case CU_MEMORYTYPE_ARRAY:
      return OS << "array";
    case CU_MEMORYTYPE_UNIFIED:
      return OS << "unified";
    default:
      return OS << "unknown";
  }
}

class ptr_attributes{
  CUcontext Ctx;
  CUmemorytype MemoryType;
  CUdeviceptr DevicePtr;
  void* HostPtr;
  bool SyncMemops;
  bool IsManaged;
  bool Mapped;
  CUdevice Device;
  uint64_t BufferId;
  CUdeviceptr RangeStartAddr;
  size_t RangeSize;
  CUmemAllocationHandleType HandleTypes;

  inline static CUpointer_attribute Attributes[]{
      CU_POINTER_ATTRIBUTE_CONTEXT,
      CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
      CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
      CU_POINTER_ATTRIBUTE_HOST_POINTER,
      CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
      CU_POINTER_ATTRIBUTE_BUFFER_ID,
      CU_POINTER_ATTRIBUTE_IS_MANAGED,
      CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
      CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
      CU_POINTER_ATTRIBUTE_RANGE_SIZE,
      CU_POINTER_ATTRIBUTE_MAPPED,
      CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
  };
public:
  explicit ptr_attributes(CUdeviceptr Ptr){
    void* Ptrs[]{
        &Ctx,
        &MemoryType,
        &DevicePtr,
        &HostPtr,
        &SyncMemops,
        &BufferId,
        &IsManaged,
        &Device,
        &RangeStartAddr,
        &RangeSize,
        &Mapped,
        &HandleTypes
    };
    cuda_assert((cuPointerGetAttributes)(std::size(Attributes), Attributes, Ptrs, Ptr))
  }

  [[nodiscard]] CUcontext context() const noexcept{
    return Ctx;
  }
  [[nodiscard]] CUmemorytype memory_type() const noexcept{
    return MemoryType;
  }
  [[nodiscard]] CUdeviceptr device_address() const noexcept{
    return DevicePtr;
  }
  [[nodiscard]] void* host_address() const noexcept{
    return HostPtr;
  }
  [[nodiscard]] bool sync_mem_ops() const noexcept{
    return SyncMemops;
  }
  [[nodiscard]] uint64_t buffer_id() const noexcept{
    return BufferId;
  }
  [[nodiscard]] bool managed() const noexcept{
    return IsManaged;
  }
  [[nodiscard]] bool mapped() const noexcept{
    return Mapped;
  }
  [[nodiscard]] CUdevice device() const noexcept{
    return Device;
  }
  [[nodiscard]] memory_range range() const noexcept{
    return {RangeStartAddr, RangeSize};
  }
  [[nodiscard]] CUmemAllocationHandleType allowed_handle_types() const noexcept{
    return HandleTypes;
  }

  friend std::ostream& operator<<(std::ostream& OS, const ptr_attributes& A){
    OS << std::boolalpha;
    OS << "{\n\tdevice_address: " << (void*)A.DevicePtr << ",\n\thost_address: " << A.HostPtr;
    OS << ",\n\n\tcontext: " << A.Ctx << ",\n\tdevice: " << A.Device << ",\n\n\t";
    OS << "memory_type: " << A.MemoryType << ",\n\tenabled_handles: "<< A.HandleTypes << ",\n\n\t";
    OS << "buffer_id: " << A.BufferId << ",\n\trange: {\n\t\taddress: " << (void*)A.RangeStartAddr << ",\n\t\t";
    OS << "size: " << A.RangeSize << "\n\t},\n\n\t" << "mapped: " << A.Mapped << ",\n\t";
    OS << "managed: " << A.IsManaged <<",\n\tmemory_sync: " << A.SyncMemops << "\n}";
    return OS;
  }
};



inline static constexpr size_t align_to(size_t Size, size_t Align){
  return Size & (Align - 1) ? (Size | (Align - 1)) + 1 : Size;
}


__global__ void twiddle_thumbs(char* Message){
  uint32_t Times = 100;
  while(--Times)
    asm volatile("nanosleep.u32 400000000;");
  if(threadIdx.x == 0){
    unsigned I = 0;
    for(char C : "Hello Maxwell Iverson, this is your GPU. I've slept for 4 seconds now, are you happy??"){
      Message[I++] = C;
    }
  }
}

int main(int argc, char** argv) {

  CUmemGenericAllocationHandle Handle;
  CUdeviceptr Address;
  size_t RangeSize;
  size_t Alignment;
  CUdevice Device;
  CUcontext Ctx;
  void* Ptr;

  CUmemAllocationProp AllocationProps;
  AllocationProps.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  AllocationProps.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  AllocationProps.win32HandleMetaData = nullptr;
  AllocationProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  AllocationProps.location.id = 0;
  AllocationProps.allocFlags = {};

  if(argc == 2)
    RangeSize = std::strtoull(argv[1], nullptr, 0);
  else
    std::cin >> RangeSize;


  unsigned ApiVersion;
  int UnifiedAddressing;

  cuda_assert((cuInit)(0))
  cuda_assert((cuDeviceGet)(&Device, 0))
  cuda_assert((cuDevicePrimaryCtxRetain)(&Ctx, Device))
  cuda_assert((cuCtxGetApiVersion)(Ctx, &ApiVersion))
  cuda_assert((cuMemGetAllocationGranularity)(&Alignment, &AllocationProps, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
  cuda_assert((cuDeviceGetAttribute)(&UnifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, Device))
  cuda_assert((cuCtxPushCurrent)(Ctx))

  std::cout << std::boolalpha;
  std::cout << "Api Version: " << ApiVersion << std::endl;
  std::cout << "Unified Addressing: " << (bool)UnifiedAddressing << std::endl;



  RangeSize = align_to(RangeSize, Alignment);

  cuda_assert((cuMemCreate)(&Handle, RangeSize, &AllocationProps, 0))
  cuda_assert((cuMemAddressReserve)(&Address, RangeSize, Alignment, 0, 0))
  cuda_assert((cuMemMap)(Address, RangeSize, 0, Handle, 0))

  std::cout << "No Access: \n";
  std::cout << ptr_attributes(Address) << std::endl;

  CUmemAccessDesc DeviceAccess;
  DeviceAccess.location = AllocationProps.location;
  DeviceAccess.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  cuda_assert((cuMemSetAccess)(Address, RangeSize, &DeviceAccess, 1))

  std::cout << "\n\nDevice access: \n";
  std::cout << ptr_attributes(Address) << std::endl;


  /*DeviceAccess.location.id = CU_DEVICE_CPU;
  DeviceAccess.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;

  cuda_assert((cuMemSetAccess)(Address, RangeSize, &DeviceAccess, 1))

  std::cout << "\n\nHost access: \n";
  std::cout << ptr_attributes(Address) << std::endl;*/
  /*CUmemAccessDesc BothAccess[2];
  BothAccess[0] = DeviceAccess;
  BothAccess[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  BothAccess[1].location.id = Device;
  BothAccess[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  cuda_assert((cuMemSetAccess)(Address, RangeSize, BothAccess, std::size(BothAccess)))

  std::cout << "\n\nBoth access: \n";
  std::cout << ptr_attributes(Address) << std::endl;*/


  CUdeviceptr UnifiedDeviceMemory;
  cuda_assert((cuMemAlloc)(&UnifiedDeviceMemory, RangeSize))

  std::cout << "\n\nUnifiedDeviceMemory: \n";
  std::cout << ptr_attributes(UnifiedDeviceMemory) << std::endl;

  CUdeviceptr UnifiedHostMemory;
  cuda_assert((cuMemHostAlloc)((void**)&UnifiedHostMemory, RangeSize, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE))

  std::cout << "\n\nUnifiedHostMemory: \n";
  std::cout << ptr_attributes(UnifiedHostMemory) << std::endl;

  CUdeviceptr GlobalManagedMemory;
  cuda_assert((cuMemAllocManaged)(&GlobalManagedMemory, RangeSize, CU_MEM_ATTACH_GLOBAL))
  CUdeviceptr HostManagedMemory;
  cuda_assert((cuMemAllocManaged)(&HostManagedMemory, RangeSize, CU_MEM_ATTACH_HOST))


  std::cout << "\n\nGlobalManagedMemory: \n";
  std::cout << ptr_attributes(GlobalManagedMemory) << std::endl;
  std::cout << "\n\nHostManagedMemory: \n";
  std::cout << ptr_attributes(HostManagedMemory) << std::endl;


  CUdeviceptr RegisteredHostMem = (CUdeviceptr)malloc(RangeSize);
  assert(RegisteredHostMem);
  cuda_assert((cuMemHostRegister)((void*)RegisteredHostMem, RangeSize, CU_MEMHOSTREGISTER_DEVICEMAP))


  std::cout << "\n\nRegisteredHostMemory: \n";
  std::cout << ptr_attributes(RegisteredHostMem) << std::endl;

  //
  {
    CUstream Stream;
    cuda_assert((cuStreamCreate)(&Stream, CU_STREAM_NON_BLOCKING))

    cuda_assert((cuLaunchHostFunc)(Stream, [](void* pUserData){
              const char Message[] = "Hello, my name is Maxwell Iverson and I am your captain for this flight, I do hope we can become good friends soon.";
              std::copy(std::execution::par_unseq, std::begin(Message), std::end(Message), (char*)pUserData);
    }, (void*)RegisteredHostMem))
    cuda_assert((cuMemcpyAsync)(UnifiedDeviceMemory, RegisteredHostMem, RangeSize, Stream))
    twiddle_thumbs<<<1, 1, 0, Stream>>>((char*)UnifiedDeviceMemory);
    cuda_assert((cuMemcpyAsync)(RegisteredHostMem, UnifiedDeviceMemory, RangeSize, Stream))
    std::cout << (const char*)RegisteredHostMem << "\n\n";
    cu::sync(Stream);
    std::cout << (const char*)RegisteredHostMem << "\n\n";

    cuda_assert((cuStreamDestroy)(Stream))
  }



  cuda_assert((cuMemUnmap)(Address, RangeSize))
  cuda_assert((cuMemAddressFree)(Address, RangeSize))
  cuda_assert((cuMemRelease)(Handle))

  cuda_assert((cuMemFree)(UnifiedDeviceMemory))
  cuda_assert((cuMemFree)(GlobalManagedMemory))
  cuda_assert((cuMemFree)(HostManagedMemory))
  cuda_assert((cuMemFreeHost)((void*)UnifiedHostMemory))
  cuda_assert((cuMemHostUnregister)((void*)RegisteredHostMem))

  cuda_assert((cuCtxPopCurrent)(&Ctx))
  cuda_assert((cuDevicePrimaryCtxRelease)(Device))


  free((void*)RegisteredHostMem);

  //print_hello_world<<<1, 64>>>();
  //cudaDeviceSynchronize();
  return 0;
}
