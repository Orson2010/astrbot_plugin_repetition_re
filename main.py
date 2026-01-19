from astrbot.api.event.filter import event_message_type, EventMessageType, command
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import *
# 导入核心消息组件以使用fromBase64方法
import astrbot.core.message.components as CoreComponents
from collections import defaultdict
from typing import List
import random
import re
import io

# 尝试导入aiohttp，如果失败则设置为None
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

# 尝试导入PIL，如果失败则设置为None
try:
    from PIL import Image as PILImage, ImageSequence, ImageChops
    PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    ImageSequence = None
    PIL_AVAILABLE = False

# 调试标志
DEBUG = False

@register("astrbot_plugin_repetition_re", "Aug", "增强版复读机插件", "1.1.0", "https://github.com/FengYing1314/astrbot_plugin_repetition")
class RepetitionPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.last_messages = defaultdict(list)
        self.repeat_count = defaultdict(int)
        self.disabled_groups = set()
        self.config = config
        
    @command("repetition")
    async def handle_repetition(self, event: AstrMessageEvent, operation: str = ""):
        '''repetition 命令处理
        
        Args:
            operation(string): on/off 开启或关闭复读功能
        '''
        if not event.message_obj.group_id:
            yield event.plain_result("此命令仅在群聊中可用")
            return
            
        if operation == "off":
            self.disabled_groups.add(event.message_obj.group_id)
            yield event.plain_result("已在本群关闭复读功能")
        elif operation == "on":
            self.disabled_groups.discard(event.message_obj.group_id)
            yield event.plain_result("已在本群开启复读功能")
        else:
            yield event.plain_result("用法: /repetition [on|off]")

    def get_message_identifier(self, message) -> str:
        """获取消息的唯一标识符"""
        result = []
        for msg in message:
            if isinstance(msg, Image):
                # 对于图片消息，使用文件名作为标识符
                result.append(f"image:{msg.file}")
            elif isinstance(msg, Plain):
                # 对于纯文本消息，使用text属性获取纯文本内容
                if hasattr(msg, 'text'):
                    result.append(msg.text)
                else:
                    result.append(str(msg))
            else:
                # 对于其他类型的消息，使用字符串表示
                result.append(str(msg))
        return "".join(result)

    async def process_image(self, image_url: str) -> str:
        """处理图片：翻转或缩放

        Args:
            image_url: 图片URL

        Returns:
            处理后的图片URL或base64数据URI
        """
        # 检查必要的库是否可用
        if not PIL_AVAILABLE or not AIOHTTP_AVAILABLE:
            if DEBUG:
                print(f"[DEBUG] PIL或aiohttp不可用，跳过图片处理: PIL={PIL_AVAILABLE}, aiohttp={AIOHTTP_AVAILABLE}")
            return image_url

        try:
            # 获取配置概率
            image_process_prob = self.config.get('image_process_probability', 0.3)
            flip_h_prob = self.config.get('flip_horizontal_probability', 0.1)
            flip_v_prob = self.config.get('flip_vertical_probability', 0.1)
            scale_prob = self.config.get('scale_probability', 0.1)
            invert_prob = self.config.get('invert_color_probability', 0.05)
            gif_frame_change_prob = self.config.get('gif_frame_change_probability', 0.1)
            gif_frame_change_min_rate = self.config.get('gif_frame_change_min_rate', 0.5)
            gif_frame_change_max_rate = self.config.get('gif_frame_change_max_rate', 2.0)
            min_scale_threshold = self.config.get('min_scale_threshold', 0.1)

            if DEBUG:
                print(f"[DEBUG] 图片处理概率配置: process={image_process_prob}, flip_h={flip_h_prob}, flip_v={flip_v_prob}, scale={scale_prob}, invert={invert_prob}, gif_frame_change={gif_frame_change_prob}")

            # 如果不需要处理图片，直接返回原URL
            rand_check = random.random()
            if rand_check >= image_process_prob:
                if DEBUG:
                    print(f"[DEBUG] 跳过图片处理，随机数{rand_check} >= {image_process_prob}")
                return image_url

            if DEBUG:
                print(f"[DEBUG] 开始处理图片: {image_url}")

            # 下载图片
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        if DEBUG:
                            print(f"[DEBUG] 图片下载失败，状态码: {response.status}")
                        return image_url
                    image_data = await response.read()

            # 打开图片
            img = PILImage.open(io.BytesIO(image_data))
            original_width, original_height = img.size
            img_format = img.format  # 获取原始图片格式

            if DEBUG:
                print(f"[DEBUG] 图片原始尺寸: {original_width}x{original_height}, 格式: {img_format}")

            # 随机选择处理方式
            rand = random.random()
            total_prob = flip_h_prob + flip_v_prob + scale_prob + invert_prob

            if total_prob > 0:
                # 标准化概率
                flip_h_prob /= total_prob
                flip_v_prob /= total_prob
                scale_prob /= total_prob
                invert_prob /= total_prob

                # 确定操作类型和参数
                operation_type = None
                operation_params = {}

                if rand < flip_h_prob:
                    # 水平翻转
                    operation_type = 'flip_horizontal'
                    if DEBUG:
                        print("[DEBUG] 执行水平翻转")
                elif rand < flip_h_prob + flip_v_prob:
                    # 垂直翻转
                    operation_type = 'flip_vertical'
                    if DEBUG:
                        print("[DEBUG] 执行垂直翻转")
                elif rand < flip_h_prob + flip_v_prob + invert_prob:
                    # 反色
                    operation_type = 'invert_color'
                    if DEBUG:
                        print("[DEBUG] 执行反色处理")
                else:
                    # 缩放
                    operation_type = 'scale'
                    min_scale = self.config.get('min_scale_percent', 0.5)
                    max_scale = self.config.get('max_scale_percent', 1.5)
                    preserve_aspect = self.config.get('preserve_aspect_ratio', True)

                    # 确保缩放比例合理
                    min_scale = max(0.01, min_scale)  # 缩放比例至少1%
                    max_scale = max(min_scale, max_scale)

                    if preserve_aspect:
                        # 等比缩放
                        # 循环随机选择缩放比例，直到满足最小变化率要求
                        max_attempts = 100  # 防止无限循环
                        scale_factor = 1.0

                        for attempt in range(max_attempts):
                            scale_factor = random.uniform(min_scale, max_scale)
                            change_rate = abs(scale_factor - 1)
                            if min_scale_threshold <= 0 or change_rate >= min_scale_threshold:
                                break
                            if DEBUG and attempt == max_attempts - 1:
                                print(f"[DEBUG] 达到最大尝试次数 {max_attempts}，使用当前缩放比例 {scale_factor:.2f}x (变化率: {change_rate:.2f})")

                        new_width = int(original_width * scale_factor)
                        new_height = int(original_height * scale_factor)
                        operation_params = {
                            'type': 'uniform',
                            'scale_factor': scale_factor,
                            'new_width': new_width,
                            'new_height': new_height
                        }
                        if DEBUG:
                            change_rate = abs(scale_factor - 1)
                            print(f"[DEBUG] 执行等比缩放 {scale_factor:.2f}x (变化率: {change_rate:.2f}), 新尺寸: {new_width}x{new_height}")
                    else:
                        # 独立缩放
                        # 为宽度和高度分别循环随机选择缩放比例，直到满足最小变化率要求
                        max_attempts = 100  # 防止无限循环
                        scale_width = 1.0
                        scale_height = 1.0

                        for attempt in range(max_attempts):
                            scale_width = random.uniform(min_scale, max_scale)
                            scale_height = random.uniform(min_scale, max_scale)
                            change_rate_width = abs(scale_width - 1)
                            change_rate_height = abs(scale_height - 1)
                            # 至少一个方向的变化率需要满足阈值要求
                            if min_scale_threshold <= 0 or change_rate_width >= min_scale_threshold or change_rate_height >= min_scale_threshold:
                                break
                            if DEBUG and attempt == max_attempts - 1:
                                print(f"[DEBUG] 达到最大尝试次数 {max_attempts}，使用当前缩放比例 {scale_width:.2f}x{scale_height:.2f} (宽度变化率: {change_rate_width:.2f}, 高度变化率: {change_rate_height:.2f})")

                        new_width = int(original_width * scale_width)
                        new_height = int(original_height * scale_height)
                        operation_params = {
                            'type': 'independent',
                            'scale_width': scale_width,
                            'scale_height': scale_height,
                            'new_width': new_width,
                            'new_height': new_height
                        }
                        if DEBUG:
                            change_rate_width = abs(scale_width - 1)
                            change_rate_height = abs(scale_height - 1)
                            print(f"[DEBUG] 执行独立缩放 {scale_width:.2f}x{scale_height:.2f} (宽度变化率: {change_rate_width:.2f}, 高度变化率: {change_rate_height:.2f}), 新尺寸: {new_width}x{new_height}")

                # 定义处理函数
                def process_single_frame(frame):
                    """处理单个图片帧"""
                    # 保存原始模式
                    original_mode = frame.mode

                    if DEBUG:
                        print(f"[DEBUG] 处理帧，原始模式: {original_mode}")

                    # 处理帧
                    if operation_type == 'flip_horizontal':
                        processed = frame.transpose(PILImage.FLIP_LEFT_RIGHT)
                    elif operation_type == 'flip_vertical':
                        processed = frame.transpose(PILImage.FLIP_TOP_BOTTOM)
                    elif operation_type == 'invert_color':
                        # 反色处理
                        # 根据图像模式进行适当转换
                        if frame.mode == 'L':
                            # 灰度模式，直接反色
                            processed = ImageChops.invert(frame)
                        elif frame.mode == 'RGB':
                            # RGB模式，直接反色
                            processed = ImageChops.invert(frame)
                        elif frame.mode == 'RGBA':
                            # RGBA模式：丢弃透明度，转换为RGB后反色
                            # 创建白色背景，将RGBA转换为RGB
                            background = PILImage.new('RGB', frame.size, (255, 255, 255))
                            background.paste(frame, mask=frame.split()[3])
                            processed = ImageChops.invert(background)
                        else:
                            # 其他模式转换为RGB后反色
                            rgb_frame = frame.convert('RGB')
                            processed = ImageChops.invert(rgb_frame)
                    elif operation_type == 'scale':
                        new_width = operation_params['new_width']
                        new_height = operation_params['new_height']
                        processed = frame.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                    else:
                        # 未知操作，返回原帧
                        return frame

                    # 简化处理：不进行调色板转换，由外部GIF处理循环统一处理
                    # 根据用户要求，可以丢失透明度，外部循环会处理模式转换

                    return processed

                # 处理图片
                if img_format == 'GIF' and ImageSequence:
                    if DEBUG:
                        print("[DEBUG] 检测到GIF格式，开始处理动图")

                    # 处理GIF动图的所有帧
                    frames = []
                    durations = []

                    for idx, frame in enumerate(ImageSequence.Iterator(img)):
                        if DEBUG:
                            print(f"[DEBUG] 处理GIF第 {idx+1} 帧，模式: {frame.mode}")

                        # 保存帧持续时间
                        durations.append(frame.info.get('duration', 100))

                        # 处理帧
                        processed_frame = process_single_frame(frame)

                        # 简化处理：转换为RGB或RGBA模式，不保留调色板
                        # 根据用户要求，可以丢失透明度
                        if processed_frame.mode in ['P', 'PA', 'RGBA', 'LA']:
                            # 如果有透明度通道，转换为RGB丢弃透明度
                            if processed_frame.mode in ['RGBA', 'LA', 'PA']:
                                if DEBUG:
                                    print(f"[DEBUG] 转换帧 {idx+1} 从 {processed_frame.mode} 到 RGB (丢弃透明度)")
                                # 创建白色背景，将RGBA转换为RGB
                                if processed_frame.mode == 'RGBA':
                                    background = PILImage.new('RGB', processed_frame.size, (255, 255, 255))
                                    background.paste(processed_frame, mask=processed_frame.split()[3])
                                    processed_frame = background
                                elif processed_frame.mode == 'LA':
                                    # LA模式：亮度+透明度
                                    background = PILImage.new('L', processed_frame.size, 255)
                                    background.paste(processed_frame, mask=processed_frame.split()[1])
                                    processed_frame = PILImage.merge('RGB', [background, background, background])
                                elif processed_frame.mode == 'PA':
                                    # PA模式：调色板+透明度，先转换为RGBA再处理
                                    rgba_frame = processed_frame.convert('RGBA')
                                    background = PILImage.new('RGB', rgba_frame.size, (255, 255, 255))
                                    background.paste(rgba_frame, mask=rgba_frame.split()[3])
                                    processed_frame = background
                            else:
                                # P模式（调色板无透明度），直接转换为RGB
                                if DEBUG:
                                    print(f"[DEBUG] 转换帧 {idx+1} 从 {processed_frame.mode} 到 RGB")
                                processed_frame = processed_frame.convert('RGB')
                        elif processed_frame.mode != 'RGB':
                            # 其他模式转换为RGB（包括L、CMYK等）
                            if DEBUG:
                                print(f"[DEBUG] 转换帧 {idx+1} 从 {processed_frame.mode} 到 RGB")
                            processed_frame = processed_frame.convert('RGB')

                        frames.append(processed_frame)

                    # 应用GIF帧数改变（调整播放速度）
                    if random.random() < gif_frame_change_prob and durations and len(frames) > 1:
                        # 保存原始帧数和持续时间用于日志
                        original_frame_count = len(frames)
                        original_total_time = sum(durations)

                        # 随机选择加速或减速比例
                        change_rate = random.uniform(gif_frame_change_min_rate, gif_frame_change_max_rate)
                        if DEBUG:
                            print(f"[DEBUG] 应用GIF帧数改变，目标速度变化: {change_rate:.2f}x (原始: {original_frame_count} 帧, 总时间: {original_total_time}ms)")

                        if change_rate < 0.8:
                            # 显著加速：通过抽帧减少帧数 (change_rate < 0.8)
                            # 计算需要保留的帧比例，确保至少保留2帧
                            keep_ratio = max(0.3, change_rate)  # 至少保留30%的帧
                            new_frame_count = max(2, int(original_frame_count * keep_ratio))

                            if DEBUG:
                                print(f"[DEBUG] 加速处理: {original_frame_count} -> {new_frame_count} 帧 (保留比例: {keep_ratio:.2f})")

                            # 智能抽帧：等间隔选择，但确保第一帧和最后一帧被保留
                            new_frames = []
                            new_durations = []

                            # 总是保留第一帧
                            new_frames.append(frames[0])
                            new_durations.append(durations[0])

                            if new_frame_count > 2:
                                # 等间隔选择中间帧
                                step = (original_frame_count - 1) / (new_frame_count - 1)
                                for i in range(1, new_frame_count - 1):
                                    idx = int(i * step)
                                    if idx >= original_frame_count:
                                        idx = original_frame_count - 1
                                    new_frames.append(frames[idx])
                                    new_durations.append(durations[idx])

                            # 总是保留最后一帧（如果不止一帧）
                            if new_frame_count > 1:
                                new_frames.append(frames[-1])
                                new_durations.append(durations[-1])

                            frames = new_frames
                            durations = new_durations

                        elif change_rate > 1.2:
                            # 显著减速：通过重复帧增加帧数 (change_rate > 1.2)
                            # 计算目标帧数，限制最大帧数
                            target_frame_count = min(100, int(original_frame_count * change_rate))

                            if DEBUG:
                                print(f"[DEBUG] 减速处理: {original_frame_count} -> {target_frame_count} 帧 (增加比例: {change_rate:.2f})")

                            # 智能重复帧：使用线性插值方法
                            new_frames = []
                            new_durations = []

                            # 计算插值步长
                            step = (original_frame_count - 1) / (target_frame_count - 1) if target_frame_count > 1 else 0

                            for i in range(target_frame_count):
                                # 计算在原始帧中的位置
                                pos = i * step
                                idx1 = int(pos)
                                idx2 = min(idx1 + 1, original_frame_count - 1)

                                if idx1 == idx2 or idx2 >= original_frame_count:
                                    # 直接使用帧
                                    new_frames.append(frames[idx1])
                                    # 调整duration以保持总时间
                                    new_duration = max(10, int(durations[idx1] / change_rate))
                                    new_durations.append(new_duration)
                                else:
                                    # 在两个帧之间，使用前一个帧（简单实现）
                                    # 更复杂的实现可以混合帧，但这里保持简单
                                    new_frames.append(frames[idx1])
                                    # 调整duration
                                    new_duration = max(10, int(durations[idx1] / change_rate))
                                    new_durations.append(new_duration)

                            frames = new_frames
                            durations = new_durations
                        else:
                            # 轻微变化 (0.8 <= change_rate <= 1.2)：使用原始duration调整方法
                            if DEBUG:
                                print(f"[DEBUG] 轻微速度调整 ({change_rate:.2f}x)，使用duration调整方法")

                            new_durations = []
                            for duration in durations:
                                new_duration = int(duration * change_rate)
                                new_duration = max(10, new_duration)  # 防止过小
                                new_durations.append(new_duration)
                            durations = new_durations

                        # 计算调整后的总时间
                        new_total_time = sum(durations)
                        if DEBUG:
                            print(f"[DEBUG] 帧数调整完成: {original_frame_count} 帧 -> {len(frames)} 帧")
                            print(f"[DEBUG] 时间调整: {original_total_time}ms -> {new_total_time}ms (速度变化: {original_total_time/new_total_time:.2f}x)")
                    elif random.random() < gif_frame_change_prob and durations and len(frames) <= 1:
                        if DEBUG:
                            print(f"[DEBUG] GIF帧数过少 ({len(frames)} 帧)，跳过帧数改变")

                    # 保存处理后的GIF
                    output_buffer = io.BytesIO()
                    if frames:
                        if DEBUG:
                            print(f"[DEBUG] 保存GIF，使用 {len(frames)} 帧，时长: {durations}")

                        try:
                            # 保存第一帧，让PIL自动处理调色板转换
                            # 不使用透明度参数，简化处理
                            frames[0].save(
                                output_buffer,
                                format='GIF',
                                save_all=True,
                                append_images=frames[1:],
                                duration=durations,
                                loop=0,  # 无限循环
                                optimize=True
                            )
                            if DEBUG:
                                print(f"[DEBUG] GIF保存成功，大小: {output_buffer.tell()} 字节")
                        except Exception as e:
                            if DEBUG:
                                print(f"[DEBUG] GIF保存失败: {e}")
                            # 回退到简单保存，不设置任何透明度
                            try:
                                frames[0].save(
                                    output_buffer,
                                    format='GIF',
                                    save_all=True,
                                    append_images=frames[1:],
                                    duration=durations,
                                    loop=0
                                )
                            except Exception as e2:
                                if DEBUG:
                                    print(f"[DEBUG] GIF保存完全失败: {e2}")
                                return image_url
                    else:
                        if DEBUG:
                            print("[DEBUG] 没有可保存的帧")
                        return image_url

                    img_data = output_buffer.getvalue()
                    mime_type = 'image/gif'
                else:
                    # 处理静态图片
                    processed_img = process_single_frame(img)
                    output_buffer = io.BytesIO()
                    # 使用原始格式保存，如果未知则使用PNG
                    save_format = img_format if img_format in ['JPEG', 'PNG', 'WEBP', 'BMP'] else 'PNG'
                    processed_img.save(output_buffer, format=save_format)
                    img_data = output_buffer.getvalue()
                    mime_type = f'image/{save_format.lower()}'
            else:
                if DEBUG:
                    print("[DEBUG] 所有处理概率为0，跳过处理")
                return image_url

            # 创建base64数据URI
            import base64
            base64_str = base64.b64encode(img_data).decode('utf-8')
            result = f"data:{mime_type};base64,{base64_str}"

            if DEBUG:
                print(f"[DEBUG] 图片处理完成，返回base64数据URI，长度: {len(result)}")

            return result

        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] 图片处理失败: {e}")
            # 如果处理失败，返回原URL
            return image_url

    async def rebuild_message_chain(self, message) -> List:
        """重建消息链，确保图片等媒体消息能正确发送"""
        return await self.process_all_images_in_chain(message)

    async def process_all_images_in_chain(self, message_chain) -> List:
        """处理消息链中的所有图片

        Args:
            message_chain: 原始消息链

        Returns:
            处理后的消息链
        """
        new_chain = []
        for msg in message_chain:
            if isinstance(msg, Image):
                # 处理图片
                processed_result = await self.process_image(msg.url)

                # 判断返回的是base64数据URI还是普通URL
                if processed_result.startswith('data:image/'):
                    # 提取base64部分
                    # 格式: data:image/png;base64,<base64_data>
                    if ';base64,' in processed_result:
                        base64_data = processed_result.split(';base64,', 1)[1]
                        if DEBUG:
                            print(f"[DEBUG] process_all_images_in_chain: 使用base64图片数据，长度: {len(base64_data)}")
                        # 使用核心组件的fromBase64方法
                        if DEBUG:
                            print(f"[DEBUG] process_all_images_in_chain: 使用CoreComponents.Image.fromBase64")
                        new_chain.append(CoreComponents.Image.fromBase64(base64_data))
                    else:
                        # 不是标准的base64数据URI，直接使用fromURL
                        if DEBUG:
                            print(f"[DEBUG] process_all_images_in_chain: 非标准base64数据URI，使用fromURL")
                        new_chain.append(Image.fromURL(processed_result))
                else:
                    # 普通URL
                    if DEBUG:
                        print(f"[DEBUG] process_all_images_in_chain: 使用普通URL: {processed_result[:50]}...")
                    new_chain.append(Image.fromURL(processed_result))
            else:
                new_chain.append(msg)
        return new_chain

    def process_text_in_chain(self, message_chain, process_func):
        """处理消息链中的文本组件

        Args:
            message_chain: 原始消息链
            process_func: 处理文本的函数，接受文本字符串，返回处理后的字符串

        Returns:
            处理后的新消息链
        """
        new_chain = []
        for msg in message_chain:
            if isinstance(msg, Plain):
                # 处理文本组件，使用text属性获取纯文本内容
                if hasattr(msg, 'text'):
                    # 如果有text属性，使用它
                    text_content = msg.text
                else:
                    # 否则使用字符串表示
                    text_content = str(msg)
                processed_text = process_func(text_content)
                new_chain.append(Plain(processed_text))
            else:
                # 非文本组件保持不变
                new_chain.append(msg)
        return new_chain

    async def reverse_message_chain(self, message_chain):
        """反转消息链中的文本内容，并处理图片"""
        def reverse_text(text):
            return text[::-1]
        # 先处理文本
        text_processed_chain = self.process_text_in_chain(message_chain, reverse_text)
        # 再处理图片
        return await self.process_all_images_in_chain(text_processed_chain)

    async def reorder_message_chain(self, message_chain):
        """随机重新排序消息链中的文本分块，并处理图片"""
        # 获取配置
        min_chunk = self.config.get('min_chunk_size', 2)
        max_chunk = self.config.get('max_chunk_size', 5)
        strength = self.config.get('reorder_strength', 0.5)
        per_chunk_times = self.config.get('reorder_per_chunk_times', 1)

        # 确保配置有效
        min_chunk = max(1, min_chunk)  # 至少1个字符
        max_chunk = max(min_chunk, max_chunk)  # 确保最大不小于最小
        strength = max(0.0, min(1.0, strength))
        per_chunk_times = max(1, per_chunk_times)

        def reorder_text(text):
            if not text:
                return text

            # 将文本转换为字符列表以便修改
            chars = list(text)
            text_len = len(chars)
            i = 0

            # 遍历文本，按随机大小的分块进行处理
            while i < text_len:
                # 随机选择分块大小，在min_chunk和max_chunk之间
                chunk_size = random.randint(min_chunk, max_chunk)
                # 确保不会超出文本边界
                end = min(i + chunk_size, text_len)

                # 如果分块太小（比如只剩1个字符），则跳过
                if end - i < 2:
                    i = end
                    continue

                chunk_len = end - i

                # 对每个分块进行多次重新排序
                for _ in range(per_chunk_times):
                    # 计算实际交换次数基于强度
                    # 强度为1时，每个分块完全随机排序；强度为0时，不排序
                    swap_count = int(strength * chunk_len * 0.5)  # 粗略估计
                    swap_count = max(1, swap_count)  # 至少交换一次

                    # 在分块内随机交换字符
                    chunk_chars = chars[i:end]
                    for _ in range(swap_count):
                        idx_i = random.randint(0, chunk_len - 1)
                        idx_j = random.randint(0, chunk_len - 1)
                        chunk_chars[idx_i], chunk_chars[idx_j] = chunk_chars[idx_j], chunk_chars[idx_i]

                    # 将修改后的字符放回原处
                    chars[i:end] = chunk_chars

                i = end

            return ''.join(chars)

        # 先处理文本
        text_processed_chain = self.process_text_in_chain(message_chain, reorder_text)
        # 再处理图片
        return await self.process_all_images_in_chain(text_processed_chain)

    @event_message_type(EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        '''自动复读相同的消息'''
        if DEBUG:
            print(f"[DEBUG] on_message: 收到消息，内容: {event.message_obj.message_str[:50]}...")

        if "/" in event.message_obj.message_str:
            if DEBUG:
                print("[DEBUG] on_message: 消息包含'/'，跳过处理")
            return
            
        # 检查群是否已禁用
        if event.message_obj.group_id and event.message_obj.group_id in self.disabled_groups:
            if DEBUG:
                print(f"[DEBUG] on_message: 群 {event.message_obj.group_id} 已禁用复读，跳过处理")
            return
            
        session_id = event.unified_msg_origin
        current_message = event.message_obj.message

        if DEBUG:
            print(f"[DEBUG] on_message: session_id={session_id}, 群组={event.message_obj.group_id}")

        if not current_message:
            if DEBUG:
                print("[DEBUG] on_message: 消息内容为空，跳过处理")
            return

        message_id = self.get_message_identifier(current_message)
        if DEBUG:
            print(f"[DEBUG] on_message: 消息标识符={message_id[:50]}...")
        if message_id == self.last_messages[session_id]:
            self.repeat_count[session_id] += 1
            if DEBUG:
                print(f"[DEBUG] on_message: 检测到重复消息，重复计数={self.repeat_count[session_id]}, 上次消息标识符={self.last_messages.get(session_id, '无')[:50]}...")

            # 获取触发阈值，确保至少为2
            threshold = self.config.get('repeat_threshold', 2)
            if threshold < 2:
                threshold = 2

            if DEBUG:
                print(f"[DEBUG] on_message: 触发阈值={threshold}, 当前计数={self.repeat_count[session_id]}")

            if self.repeat_count[session_id] == threshold - 1:
                break_spell_prob = self.config.get('break_spell_probability', 0.3)
                break_spell_rand = random.random()
                if DEBUG:
                    print(f"[DEBUG] on_message: 检查打断施法，概率={break_spell_prob}, 随机数={break_spell_rand}")

                if break_spell_rand < break_spell_prob:
                    if DEBUG:
                        print("[DEBUG] on_message: 触发打断施法")
                    yield event.plain_result(self.config.get('break_spell_text', '打断施法！'))
                else:
                    # 获取倒读和重新排序的概率配置
                    reverse_prob = self.config.get('reverse_probability', 0.1)
                    reorder_prob = self.config.get('reorder_probability', 0.1)

                    if DEBUG:
                        print(f"[DEBUG] on_message: 复读类型概率配置: 倒读={reverse_prob}, 重排序={reorder_prob}")

                    # 确保概率在有效范围内
                    reverse_prob = max(0.0, min(1.0, reverse_prob))
                    reorder_prob = max(0.0, min(1.0, reorder_prob))

                    # 如果概率总和超过1，按比例缩放
                    total_prob = reverse_prob + reorder_prob
                    if total_prob > 1.0:
                        reverse_prob /= total_prob
                        reorder_prob /= total_prob
                        total_prob = 1.0

                    rand = random.random()
                    if DEBUG:
                        print(f"[DEBUG] on_message: 复读类型随机数={rand}, 总概率={total_prob}, 调整后: 倒读={reverse_prob}, 重排序={reorder_prob}")

                    if rand < reverse_prob:
                        # 触发倒读复读
                        if DEBUG:
                            print("[DEBUG] on_message: 选择倒读复读")
                        new_chain = await self.reverse_message_chain(current_message)
                    elif rand < total_prob:
                        # 触发随机重新排序复读
                        if DEBUG:
                            print("[DEBUG] on_message: 选择随机重新排序复读")
                        new_chain = await self.reorder_message_chain(current_message)
                    else:
                        # 正常复读
                        if DEBUG:
                            print("[DEBUG] on_message: 选择正常复读")
                        new_chain = await self.rebuild_message_chain(current_message)

                    if DEBUG:
                        # 检查消息链中是否有图片
                        image_count = sum(1 for msg in new_chain if isinstance(msg, Image))
                        plain_count = sum(1 for msg in new_chain if isinstance(msg, Plain))
                        print(f"[DEBUG] on_message: 处理完成，消息链包含 {len(new_chain)} 个组件 (图片: {image_count}, 文本: {plain_count})")

                    yield event.chain_result(new_chain)
        else:
            if DEBUG:
                print(f"[DEBUG] on_message: 消息不重复，重置计数器")
            self.repeat_count[session_id] = 0

        self.last_messages[session_id] = message_id
        if DEBUG:
            print(f"[DEBUG] on_message: 更新会话 {session_id} 的最后消息标识符")
