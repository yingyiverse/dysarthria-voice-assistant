# 前端开发指南 (Frontend)

## 技术栈

- **框架**: Next.js 14 (App Router)
- **UI 库**: React 18 + Tailwind CSS
- **状态管理**: Zustand
- **PWA**: next-pwa
- **音频处理**: Web Audio API

## 目录结构

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── page.tsx            # 首页/实时转录
│   │   ├── history/            # 历史记录
│   │   ├── assistant/          # 语音主持人
│   │   ├── training/           # 个性化训练
│   │   └── settings/           # 设置
│   ├── components/
│   │   ├── audio/              # 音频相关组件
│   │   │   ├── AudioRecorder.tsx
│   │   │   ├── VoiceVisualizer.tsx
│   │   │   └── StreamingPlayer.tsx
│   │   ├── transcript/         # 转录相关组件
│   │   └── ui/                 # 通用 UI 组件
│   ├── lib/
│   │   ├── api/                # API 客户端
│   │   ├── audio/              # 音频处理工具
│   │   └── websocket/          # WebSocket 客户端
│   ├── hooks/                  # React Hooks
│   └── stores/                 # Zustand 状态管理
├── public/
│   ├── manifest.json           # PWA 配置
│   └── sw.js                   # Service Worker
├── package.json
└── next.config.js
```

## 快速开始

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建
npm run build

# 生产模式
npm start
```

## 环境变量

创建 `.env.local` 文件：

```bash
# API 地址
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## 与后端交互

### HTTP API

参考 `docs/API_SPECIFICATION.md` 获取完整 API 文档。

主要 API 端点：
- `POST /api/v1/asr/transcribe` - 语音转文字
- `POST /api/v1/agent/chat` - 对话交互
- `GET /api/v1/sessions` - 获取会话列表

### WebSocket

实时转录使用 WebSocket 连接：

```typescript
const ws = new WebSocket('ws://localhost:8000/ws/asr');

// 发送音频数据
ws.send(audioBuffer);

// 接收转录结果
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(result.text);
};
```

## PWA 配置

项目使用 PWA 支持离线功能：

1. **Service Worker** - 缓存静态资源
2. **manifest.json** - 定义应用元数据
3. **离线存储** - IndexedDB 存储历史记录

## 开发规范

- 组件使用 TypeScript
- 遵循 ESLint 规则
- 使用 Prettier 格式化
- 组件测试使用 Jest + React Testing Library

## 相关文档

- [API 规范](../docs/API_SPECIFICATION.md)
- [架构设计](../docs/ARCHITECTURE.md)
- [主项目 README](../README.md)
