#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
#define OFFSET3D(x, y, z, ld1, ld2) ((x) * (ld1) * (ld2) + (y) * (ld2) + (z))

#define KPACK_b8  4
#define KPACK_b16 2
#define KPACK_b32 1

// amx tile load/store L1
#define amx_tile_load_L1A(dst, arr, row, col, ld) \
    _tile_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int8_t))
#define amx_tile_load_L1B(dst, arr, row, col, ld) \
    _tile_loadd(dst, &arr[OFFSET2D( \
        (row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)], \
        (ld) * KPACK_b8 * sizeof(int8_t))
#define amx_tile_load_L1C(dst, arr, row, col, ld) \
    _tile_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))
#define amx_tile_store_L1C(src, arr, row, col, ld) \
    _tile_stored(src, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))

// amx tile load L2
#define amx_tile_load_L2A(dst, arr, row, col, ld) \
    _tile_stream_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int8_t))
#define amx_tile_load_L2B(dst, arr, row, col, ld) \
    _tile_stream_loadd(dst, &arr[OFFSET2D( \
        (row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)], \
        (ld) * KPACK_b8 * sizeof(int8_t))
#define amx_tile_load_L2C(dst, arr, row, col, ld) \
    _tile_stream_loadd(dst, &arr[OFFSET2D(row, col, ld)], (ld) * sizeof(int32_t))

// amx tile prefetch L1
#define amx_tile_prefetch_L1A(arr, row, col, ld) \
    _mm_prefetch((const char *)&arr[OFFSET2D(row, col, ld)], _MM_HINT_T0)
#define amx_tile_prefetch_L1B(arr, row, col, ld) \
    _mm_prefetch((const char *)&arr[OFFSET2D( \
        (row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)], _MM_HINT_T0)

// amx tile prefetch L2
#define amx_tile_prefetch_L2A(arr, row, col, ld) \
    _mm_prefetch((const char *)&arr[OFFSET2D(row, col, ld)], _MM_HINT_T1)
#define amx_tile_prefetch_L2B(arr, row, col, ld) \
    _mm_prefetch((const char *)&arr[OFFSET2D( \
        (row) / KPACK_b8, (col) * KPACK_b8, (ld) * KPACK_b8)], _MM_HINT_T1)

// Define tile config data structure
typedef struct __tile_config
{
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __tilecfg;

/* Initialize tile config */
static void init_tile_config (__tilecfg *tileinfo)
{
  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  for (i = 0; i < 8; ++i)
  {
    tileinfo->colsb[i] = MAX_COLS;
    tileinfo->rows[i] =  MAX_ROWS;
  }

  _tile_loadconfig (tileinfo);
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
   {
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
      printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}